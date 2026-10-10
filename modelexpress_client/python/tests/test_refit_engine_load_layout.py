# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import ctypes
import hashlib
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from modelexpress.refit.reshard.types import IncompleteRefit
from modelexpress_rl.inference.engines.vllm import installer as module
from torch import nn


@pytest.fixture
def capture_installer(monkeypatch) -> tuple:
    names = (
        "vllm",
        "vllm.config",
        "vllm.model_executor",
        "vllm.model_executor.model_loader",
        "vllm.model_executor.model_loader.reload",
        "vllm.model_executor.model_loader.reload.layerwise",
    )
    modules = {name: ModuleType(name) for name in names}
    modules["vllm.config"].set_current_vllm_config = lambda config: nullcontext()
    layerwise = modules["vllm.model_executor.model_loader.reload.layerwise"]
    layerwise.LAYERWISE_INFO = {}
    layerwise.initialize_layerwise_reload = lambda model: None
    layerwise._get_original_loader = lambda parameter: None
    layerwise._place_kernel_tensors = lambda layer, info: None
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    weight_utils = ModuleType("vllm.model_executor.model_loader.weight_utils")
    weight_utils.default_weight_loader = lambda parameter, weight: parameter.data.copy_(
        weight
    )
    monkeypatch.setitem(sys.modules, weight_utils.__name__, weight_utils)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2, 2))
            self.capture_calls = 0

        def load_weights(self, weights) -> None:
            self.capture_calls += 1
            for name, weight in weights:
                self.weight.weight_loader(self.weight, weight[:2, :2])

    model = Model()
    installer = module._VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    return installer, model


@pytest.mark.parametrize("quantized", [False, True])
def test_layout_survives_runtime_parameter_replacement(
    capture_installer, quantized
) -> None:
    installer, model = capture_installer
    installer._vllm_config = SimpleNamespace(
        quant_config=object() if quantized else None
    )
    manifest = [("weight", torch.float32, (3, 3))]
    expected = installer.capture(manifest)

    def unavailable(weights) -> None:
        raise AssertionError("engine capture is no longer available")

    model.load_weights = unavailable
    model.weight = nn.Parameter(torch.ones(2, 2))
    first, layout = installer.capture(manifest)
    assert (first, layout) == expected
    first.copies[0].dest_offset = 55
    layout.clear()
    assert installer.capture(manifest) == expected


def test_replacement_native_source_composes_views(capture_installer) -> None:
    installer, model = capture_installer
    installer._convert_native_to_hf = lambda weights: {
        "weight": next(iter(weights.values())).transpose(0, 1)
    }
    installer.capture([("first", torch.float32, (3, 3))])

    def unavailable(weights) -> None:
        raise AssertionError("engine capture is no longer available")

    model.load_weights = unavailable
    capture, layout = installer.capture([("replacement", torch.float32, (3, 3))])
    record = capture.copies[0]
    assert record.src_name == "replacement"
    # Execute the recipe against real replacement bytes.
    source = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    value = source
    for operation, args, kwargs in record.op_chain:
        value = getattr(value, operation)(*args, **dict(kwargs))
    assert torch.equal(value, source.T[:2, :2])
    assert layout == {"weight": ((2, 2), torch.float32)}


def test_incompatible_inputs_do_not_replace_layout(capture_installer) -> None:
    installer, model = capture_installer
    manifest = [("weight", torch.float32, (3, 3))]
    expected = installer.capture(manifest)

    def unavailable(weights) -> None:
        raise AssertionError("engine capture is no longer available")

    model.load_weights = unavailable
    for invalid in (
        [("weight", torch.float32, (4, 4))],
        [("other", torch.float32, (3, 3))],
        [("weight", torch.float16, (3, 3))],
    ):
        with pytest.raises(IncompleteRefit, match="fixed engine input layout"):
            installer.capture(invalid)
    assert installer.capture(manifest) == expected


def test_failed_initial_capture_allows_later_initialization(
    capture_installer, monkeypatch
) -> None:
    installer, model = capture_installer
    original = model.load_weights

    def fail(weights) -> None:
        raise RuntimeError("capture failed")

    model.load_weights = fail
    with pytest.raises(RuntimeError, match="capture failed"):
        installer.capture([("weight", torch.float32, (3, 3))])
    model.load_weights = original
    capture, layout = installer.capture([("weight", torch.float32, (4, 4))])
    assert capture.copies and layout == {"weight": ((2, 2), torch.float32)}


def test_failed_restoration_does_not_fix_input_schema(capture_installer) -> None:
    installer, model = capture_installer
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]

    def fail() -> None:
        raise RuntimeError("restoration failed")

    layerwise.LAYERWISE_INFO[model] = SimpleNamespace(kernel_tensors=None, reset=fail)
    with pytest.raises(RuntimeError, match="restoration failed"):
        installer.capture([("weight", torch.float32, (3, 3))])
    layerwise.LAYERWISE_INFO.clear()
    capture, layout = installer.capture([("weight", torch.float32, (4, 4))])
    assert capture.copies and layout == {"weight": ((2, 2), torch.float32)}


def test_incomplete_initial_capture_can_be_retried(capture_installer) -> None:
    installer, model = capture_installer
    loader = model.load_weights
    model.load_weights = lambda weights: None
    with pytest.raises(IncompleteRefit, match="cover every parameter"):
        installer.capture([("weight", torch.float32, (3, 3))])
    model.load_weights = loader
    capture, layout = installer.capture([("weight", torch.float32, (4, 4))])
    assert capture.copies and layout == {"weight": ((2, 2), torch.float32)}


@pytest.mark.parametrize("quantized", [False, True])
def test_fixed_engine_owner_transfers_alternate_replicas_without_recapture(
    capture_installer, monkeypatch, quantized
) -> None:
    from modelexpress.refit.reshard.rendezvous import PublishedShard, PublishedTensor, wrap_rendezvous_blob
    from modelexpress_rl.inference import nixl_staged_transfer as transfer_module
    from modelexpress_rl.inference.adapter import TrainerSourceShard
    from modelexpress_rl.inference.plan import TrainerSourceSnapshot

    installer, model = capture_installer
    installer._vllm_config = SimpleNamespace(quant_config=object() if quantized else None)
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(transfer_module, "classic_cuda_alloc", nullcontext)
    manager = MagicMock()
    manager.add_remote_agent.return_value = "source"

    class Transport:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def post_reads(self, descriptors) -> list:
            for descriptor in descriptors:
                ctypes.memmove(descriptor.dst_addr, descriptor.src_addr, descriptor.nbytes)
            return []

        def await_reads(self, posted) -> None:
            pass

    monkeypatch.setattr(transfer_module, "NixlReshardTransport", Transport)
    transfer = transfer_module._NixlStagedTransfer(
        manager=manager, device_id=0, device=torch.device("cpu")
    )
    replicas = {"A": torch.arange(9, dtype=torch.float32).reshape(3, 3)}
    replicas["B"] = replicas["A"].clone() + 100
    try:
        for index, worker in enumerate(("A", "B", "A")):
            value = replicas[worker]
            blob = wrap_rendezvous_blob(b"source", "source", "source:19000", [
                PublishedTensor("weight", "torch.float32", 4, (3, 3), [
                    PublishedShard("source", 0, value.data_ptr(), (0, 0), (3, 3))
                ])
            ])
            shard = TrainerSourceShard("slot", worker, hashlib.sha256(blob).hexdigest(), "source:19000", blob)
            prepared = transfer.prepare_full_copy(
                trainer_snapshot=TrainerSourceSnapshot("mesh", 1, (shard,)),
                manifests=[blob], capture_layout=installer.capture,
            )
            assert torch.equal(transfer.stage(prepared).tensors["weight"], value[:2, :2])
            if index:
                assert prepared.metrics.get("initial_whole_plan_s", 0) == 0
            else:
                def unavailable(weights) -> None:
                    raise AssertionError("engine capture is no longer available")

                model.load_weights = unavailable
    finally:
        transfer.close()

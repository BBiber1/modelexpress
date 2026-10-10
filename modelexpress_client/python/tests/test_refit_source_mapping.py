# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from collections.abc import Iterator
from contextlib import nullcontext
from types import ModuleType

import pytest
import torch
from torch import nn
import modelexpress_rl.inference.engines.vllm.installer as module


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
            self.register_buffer("routing", torch.tensor([0, 1]))
            self.reject_capture = False

        def load_weights(self, weights) -> None:
            if self.reject_capture:
                raise RuntimeError("unexpected source capture")
            start = int(self.routing[0])
            for name, weight in weights:
                self.weight.weight_loader(self.weight, weight[start : start + 2, :2])

    model = Model()
    installer = module._VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    return installer, model


@pytest.fixture(params=[False, True])
def mapping_transfer(capture_installer, monkeypatch, request) -> Iterator[object]:
    import ctypes
    from types import SimpleNamespace
    import modelexpress_rl.inference.nixl_staged_transfer as transfer_module
    from modelexpress.refit.reshard.rendezvous import (
        PublishedShard,
        PublishedTensor,
        wrap_rendezvous_blob,
    )
    from modelexpress_rl.inference.plan import StreamingSettings, TrainerSourceSnapshot

    installer, model = capture_installer
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    monkeypatch.setenv("MX_REFIT_CACHE_BOUNDED_PLANS", "1")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(transfer_module, "classic_cuda_alloc", nullcontext)
    real_empty = torch.empty

    def empty(*args, **kwargs) -> torch.Tensor:
        if torch.device(kwargs.get("device", "cpu")).type == "cuda":
            kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    events = []

    class Manager:
        def __init__(self, **kwargs) -> None:
            self.registrations = {}

        def initialize(self) -> None:
            pass

        def shutdown(self) -> None:
            self.registrations.clear()

        def add_remote_agent(self, metadata) -> str:
            return "source"

        def register_tensors(self, tensors) -> None:
            self.registrations.update(tensors)

        def register_dram_buffer(self, tensor) -> object:
            handle = object()
            self.registrations[handle] = tensor
            return handle

        def deregister_memory(self, handle) -> None:
            del self.registrations[handle]

    class Transport:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def post_reads(self, descriptors) -> list:
            events.extend(descriptors)
            for descriptor in descriptors:
                ctypes.memmove(
                    descriptor.dst_addr, descriptor.src_addr, descriptor.nbytes
                )
            return [object()]

        def await_reads(self, posted) -> None:
            pass

    monkeypatch.setattr(transfer_module, "NixlTransferManager", Manager)
    monkeypatch.setattr(transfer_module, "NixlReshardTransport", Transport)
    transfer = transfer_module._NixlStagedTransfer(
        device_id=0,
        device=torch.device("cuda:0"),
        agent_name="target",
        streaming=StreamingSettings(512, "cpu") if request.param else None,
    )
    captured = []

    def capture(manifest) -> tuple:
        result = installer.capture(manifest)
        captured.append(result)
        return result

    def prepare(source, mesh="mesh") -> object:
        manifest = wrap_rendezvous_blob(
            b"source",
            "source",
            "source:19000",
            [
                PublishedTensor(
                    name="weight",
                    dtype="torch.float32",
                    elsize=4,
                    full_shape=tuple(source.shape),
                    shards=[
                        PublishedShard(
                            agent_name="source",
                            device_id=0,
                            addr=source.data_ptr(),
                            shard_offset=(0, 0),
                            shape=tuple(source.shape),
                        )
                    ],
                )
            ],
        )
        prepare_transfer = (
            transfer.prepare_streaming if request.param else transfer.prepare_full_copy
        )
        return prepare_transfer(
            manifests=[manifest],
            trainer_snapshot=TrainerSourceSnapshot(mesh, 1, ()),
            capture_layout=capture,
            source_mapping_key=installer._capture_key,
        )

    def receive(prepared) -> torch.Tensor:
        if request.param:
            return next(transfer.iter_bounded(prepared, {}))["weight"].clone()
        return transfer.stage(prepared).tensors["weight"].clone()

    yield SimpleNamespace(
        installer=installer,
        model=model,
        prepare=prepare,
        receive=receive,
        events=events,
        captured=captured,
        transfer=transfer,
    )
    transfer.close()


def test_compatible_physical_replacement_retains_mapping_and_reads_new_values(
    mapping_transfer,
) -> None:
    h = mapping_transfer
    first = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    prepared = h.prepare(first)
    assert torch.equal(h.receive(prepared), first[:2, :2])
    h.model.reject_capture = True
    replacement = first + 100
    prepared = h.prepare(replacement, "replacement-mesh")
    assert torch.equal(h.receive(prepared), replacement[:2, :2])
    assert any(read.src_addr == replacement.data_ptr() for read in h.events)


@pytest.mark.parametrize(
    "change", ["routing", "inference_routing", "loader", "module_loader", "conversion"]
)
def test_mapping_validity_changes_install_correct_source_slices(
    mapping_transfer, change
) -> None:
    h = mapping_transfer
    source = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    assert torch.equal(h.receive(h.prepare(source)), source[:2, :2])
    if change == "routing":
        h.model.routing.add_(1)
    elif change == "inference_routing":
        with torch.inference_mode():
            h.model.routing = torch.tensor([1, 2])
        assert torch.equal(h.receive(h.prepare(source)), source[1:3, :2])
        with torch.inference_mode():
            h.model.routing.add_(1)
    elif change == "loader":

        def replacement_loader(parameter, tensor) -> None:
            parameter.data.copy_(tensor.transpose(0, 1))

        h.installer._original_loader = lambda _: replacement_loader
    elif change == "module_loader":
        from types import MethodType

        def replacement_load(model, weights) -> None:
            for _, tensor in weights:
                model.weight.weight_loader(model.weight, tensor[1:3, :2])

        h.model.load_weights = MethodType(replacement_load, h.model)
    else:
        h.installer._convert_native_to_hf = lambda weights: {
            name: tensor.transpose(0, 1) for name, tensor in weights.items()
        }
    rows = h.model.routing.tolist()
    expected = source[rows, :2]
    if change in ("loader", "conversion"):
        expected = expected.transpose(0, 1)
    elif change == "module_loader":
        expected = source[1:3, :2]
    assert torch.equal(h.receive(h.prepare(source)), expected)


def test_caller_mutation_does_not_change_owned_mapping(mapping_transfer) -> None:
    h = mapping_transfer
    source = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    prepared = h.prepare(source)
    mapping, layout = h.captured[-1]
    record = mapping.copies[0]
    record.dest_offset = 99
    mapping.copies.clear()
    with pytest.raises(TypeError):
        layout["weight"] = ((1, 1), torch.bfloat16)
    assert torch.equal(h.receive(prepared), source[:2, :2])
    assert torch.equal(h.receive(h.prepare(source)), source[:2, :2])


@pytest.mark.parametrize("change", ["parameter", "source_schema", "quantized"])
def test_invalid_mapping_key_requires_fresh_capture_before_reads(
    mapping_transfer, change
) -> None:
    from types import SimpleNamespace

    h = mapping_transfer
    source = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    assert torch.equal(h.receive(h.prepare(source)), source[:2, :2])
    if change == "parameter":
        h.model.weight = nn.Parameter(torch.zeros(2, 2))
    elif change == "source_schema":
        source = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    else:
        h.installer._vllm_config = SimpleNamespace(quant_config=object())
    h.model.reject_capture = True
    h.events.clear()
    with pytest.raises(RuntimeError, match="unexpected source capture"):
        h.prepare(source)
    assert h.events == []

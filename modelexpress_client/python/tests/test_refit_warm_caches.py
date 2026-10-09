# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import time
from dataclasses import replace
from types import SimpleNamespace

import modelexpress_rl.inference.nixl_staged_transfer as transfer_module
import pytest
import torch
from modelexpress.refit.reshard.rendezvous import (
    PublishedShard,
    PublishedTensor,
    wrap_rendezvous_blob,
)
from modelexpress.refit.reshard.slice_plan import Shard
from modelexpress.refit.reshard.transfer_plan import (
    ConvertSource,
    SourceInfo,
    TransferPlan,
)
from modelexpress.refit.reshard.types import (
    CaptureResult,
    IncompleteRefit,
    RecordedCopy,
)
from modelexpress_rl.inference._cache_config import RefitCacheConfig
from modelexpress_rl.inference.nixl_staged_transfer import (
    _bounded_batches,
    _compile_bounded_plan,
    _NixlStagedTransfer,
    _plan_staged_transfer,
    _PreparedNixlTransfer,
    _resolve_sources,
)


def _manifest(*, agent_name: str, endpoint: str, offset: int, address: int) -> bytes:
    return wrap_rendezvous_blob(
        b"nixl-metadata",
        agent_name,
        endpoint,
        [
            PublishedTensor(
                name="weight",
                dtype="torch.float32",
                elsize=4,
                full_shape=(4,),
                shards=[
                    PublishedShard(
                        agent_name=agent_name,
                        device_id=0,
                        addr=address,
                        shard_offset=(offset,),
                        shape=(2,),
                    )
                ],
            )
        ],
    )


@pytest.mark.parametrize("padded", [False, True])
def test_converted_copy_uses_captured_slice_with_arena_storage_offset(
    monkeypatch, padded
) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    width = 6 if padded else 4
    offset = 1 if padded else 0
    arena = torch.full((8 * width + 16,), -17.0, dtype=torch.bfloat16)
    target = arena[8 : 8 + 8 * width].view(8, width)
    source = torch.arange(32, dtype=torch.float32).view(8, 4) / 1000
    capture = CaptureResult(
        copies=[
            RecordedCopy(
                "source", (), "layer.weight", offset, (8, 4), (width, 1), torch.bfloat16
            )
        ]
    )
    plan = TransferPlan(
        converts=[ConvertSource("layer.weight", (8, 4), torch.float32, [])]
    )
    transfer = object.__new__(_NixlStagedTransfer)
    transfer.cache_config = RefitCacheConfig()
    transfer._recv_buffers = {"layer.weight": target}
    transfer._convert_buffers = {"layer.weight": source}
    transfer._device = torch.device("cpu")
    transfer._device_id = 0
    prepared = _PreparedNixlTransfer(
        plan,
        capture,
        {},
        (),
        SimpleNamespace(await_reads=lambda _: None),
        {copy.param_name: copy for copy in capture.copies},
        0,
    )
    for version in range(3):
        source.add_(1)
        transfer._complete_stage(prepared, [], time.perf_counter())
        assert torch.equal(target[:, offset : offset + 4], source.to(torch.bfloat16))
        assert torch.all(arena[:8] == -17) and torch.all(arena[-8:] == -17)
        if padded:
            assert torch.all(target[:, 0] == -17) and torch.all(target[:, -1] == -17)


def _bounded_plan_inputs() -> dict[str, object]:
    manifests = [
        _manifest(agent_name="a", endpoint="a:19000", offset=0, address=100),
        _manifest(agent_name="b", endpoint="b:19000", offset=2, address=200),
    ]
    return {
        "resolved": _resolve_sources(manifests),
        "capture": CaptureResult(
            copies=[
                RecordedCopy("weight", (), "layer.weight", 0, (4,), (1,), torch.float32)
            ]
        ),
        "parameter_layout": {"layer.weight": ((4,), torch.float32)},
        "max_staging_bytes": 512,
        "cache_config": RefitCacheConfig(),
    }


@pytest.mark.parametrize(
    "defect", ["unattributed", "unsupported", "missing", "unknown", "fallback"]
)
def test_supplied_complete_plan_keeps_global_coverage_gate(monkeypatch, defect):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    source = SourceInfo((4,), torch.float32, 4, [Shard((0,), (4,), "s", 100, 4)])
    capture = CaptureResult(
        copies=[RecordedCopy("w", (), "layer.w", 0, (4,), (1,), torch.float32)]
    )
    layout = {"layer.w": ((4,), torch.float32)}
    if defect == "unattributed":
        capture.unattributed = 1
    elif defect == "unsupported":
        capture.unsupported.append("w")
    elif defect == "missing":
        layout["other.w"] = ((4,), torch.float32)
    elif defect == "unknown":
        capture.copies.append(replace(capture.copies[0], param_name="unknown.w"))
    complete = _plan_staged_transfer(capture, {"w": source})
    if defect == "fallback":
        complete.fallback.append("w")
    with pytest.raises(IncompleteRefit):
        _bounded_batches(capture, layout, {"w": source}, 512, complete_plan=complete)


@pytest.mark.parametrize("missing", ["session", "agent"])
def test_compiled_plan_requires_complete_agent_metadata(monkeypatch, missing) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    args = _bounded_plan_inputs()
    _compile_bounded_plan(**args, metrics={})
    if missing == "session":
        del args["resolved"].session_to_agent["a"]
        message = "unknown source sessions"
    else:
        del args["resolved"].agent_metadata["a"]
        message = "without NIXL metadata"
    with pytest.raises(RuntimeError, match=message):
        _compile_bounded_plan(**args, metrics={})

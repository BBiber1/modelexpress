# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import time
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

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
from modelexpress_rl.inference.nixl_staged_transfer import (
    _bounded_batches,
    _NixlStagedTransfer,
    _plan_staged_transfer,
    _PreparedNixlTransfer,
    _resolve_sources,
    _WeightUpdatePlan,
)

from modelexpress_rl.inference.adapter import TrainerSourceShard
from modelexpress_rl.inference.plan import TrainerSourceSnapshot

from tests.test_refit_bounded_descriptors import harness, _check_values


def _manifest(
    *,
    agent_name: str,
    endpoint: str,
    offset: int,
    address: int,
    name: str = "weight",
    dtype: str = "torch.float32",
    elsize: int = 4,
    full_shape: tuple[int, ...] = (4,),
    shard_shape: tuple[int, ...] = (2,),
) -> bytes:
    return wrap_rendezvous_blob(
        b"nixl-metadata",
        agent_name,
        endpoint,
        [
            PublishedTensor(
                name=name,
                dtype=dtype,
                elsize=elsize,
                full_shape=full_shape,
                shards=[
                    PublishedShard(
                        agent_name=agent_name,
                        device_id=0,
                        addr=address,
                        shard_offset=(offset,),
                        shape=shard_shape,
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
    transfer._recv_buffers = {"layer.weight": target}
    transfer._convert_buffers = {"layer.weight": source}
    transfer._device = torch.device("cpu")
    transfer._device_id = 0
    prepared = _PreparedNixlTransfer(
        plan, capture, {}, (), SimpleNamespace(await_reads=lambda _: None)
    )
    for version in range(3):
        source.add_(1)
        transfer._complete_stage(prepared, [], time.perf_counter())
        assert torch.equal(target[:, offset : offset + 4], source.to(torch.bfloat16))
        assert torch.all(arena[:8] == -17) and torch.all(arena[-8:] == -17)
        if padded:
            assert torch.all(target[:, 0] == -17) and torch.all(target[:, -1] == -17)


def _bounded_cache_inputs() -> dict:
    manifests = [
        _manifest(agent_name="a", endpoint="a:19000", offset=0, address=100),
        _manifest(agent_name="b", endpoint="b:19000", offset=2, address=200),
    ]
    return {
        "manifests": manifests,
        "resolved": _resolve_sources(manifests),
        "capture": CaptureResult(
            copies=[
                RecordedCopy("weight", (), "layer.weight", 0, (4,), (1,), torch.float32)
            ]
        ),
        "parameter_layout": {"layer.weight": ((4,), torch.float32)},
        "max_staging_bytes": 512,
    }


def _transfer_with_resolved_plan(monkeypatch, manifests: list[bytes]) -> tuple:
    transfer = object.__new__(_NixlStagedTransfer)
    transfer._weight_update_plan = None
    resolved = transfer._resolve_metadata(manifests, {})
    transfer._weight_update_plan = _WeightUpdatePlan(
        trainer_source_snapshot=TrainerSourceSnapshot(
            "mesh", 1, tuple(
                TrainerSourceShard(str(index), str(index), hashlib.sha256(blob).hexdigest(), "source:19000", blob)
                for index, blob in enumerate(manifests)
            ), resolved
        ),
        generator_capture_snapshot=CaptureResult(copies=[]),
        parameter_layout=MappingProxyType({}),
        transfer_plan=TransferPlan(),
    )
    return transfer, resolved


@pytest.mark.parametrize(
    ("name", "dtype", "elsize", "shape"),
    [
        ("renamed", "torch.float32", 4, (4,)),
        ("weight", "torch.bfloat16", 2, (4,)),
        ("weight", "torch.float32", 4, (8,)),
    ],
)
def test_layout_capture_reuse_tracks_resolved_source_schema(
    monkeypatch, name, dtype, elsize, shape
) -> None:
    manifest = _manifest(
        agent_name="a", endpoint="a:19000", offset=0, address=100, shard_shape=(4,)
    )
    transfer = object.__new__(_NixlStagedTransfer)
    transfer._weight_update_plan = None
    transfer._debug_validate_layout = False
    capture = CaptureResult(
        copies=[RecordedCopy("weight", (), "layer.weight", 0, (4,), (1,), torch.float32)]
    )
    parameter_layout = {"layer.weight": ((4,), torch.float32)}
    expected_schema = [("weight", torch.float32, (4,))]

    def capture_layout(current_manifest) -> tuple[CaptureResult, dict]:
        assert current_manifest == expected_schema
        return capture, parameter_layout

    def source_snapshot(mesh_id: str, generation: int, metadata: bytes) -> TrainerSourceSnapshot:
        return TrainerSourceSnapshot(mesh_id, generation, (TrainerSourceShard(
            "rank:0", "trainer", hashlib.sha256(metadata).hexdigest(), "a:19000", metadata,
        ),))

    trainer, actual_capture, actual_layout = transfer._resolve_layout(
        [manifest],
        capture_layout,
        {},
        source_snapshot("mesh", 1, manifest),
    )

    assert trainer.mesh_id == "mesh"
    assert actual_capture.copies == capture.copies
    assert actual_layout == parameter_layout
    cached_layout = MappingProxyType(parameter_layout)
    transfer._weight_update_plan = _WeightUpdatePlan(
        trainer_source_snapshot=trainer,
        generator_capture_snapshot=actual_capture,
        parameter_layout=cached_layout,
        transfer_plan=TransferPlan(),
    )

    same_schema = _manifest(
        agent_name="a", endpoint="a:19000", offset=0, address=300, shard_shape=(4,)
    )
    refreshed, reused_capture, reused_layout = transfer._resolve_layout(
        [same_schema],
        capture_layout,
        {},
        source_snapshot("new-mesh", 2, same_schema),
    )
    assert refreshed.mesh_id == "new-mesh"
    assert refreshed.mesh_generation == 2
    assert reused_capture.copies == capture.copies
    assert reused_layout == parameter_layout
    assert (
        refreshed.resolved_metadata.sources["weight"].shards[0].addr == 300
    )
    plan = _plan_staged_transfer(
        capture, refreshed.resolved_metadata.sources
    )
    assert plan.segments[0].src_addr == 300

    changed_schema = _manifest(
        agent_name="a",
        endpoint="a:19000",
        offset=0,
        address=300,
        name=name,
        dtype=dtype,
        elsize=elsize,
        full_shape=shape,
    )
    expected_schema = [(name, getattr(torch, dtype.split(".")[-1]), shape)]
    changed, recaptured, recaptured_layout = transfer._resolve_layout(
        [changed_schema],
        capture_layout,
        {},
        source_snapshot("new-mesh", 2, changed_schema),
    )
    assert recaptured.copies == capture.copies
    assert recaptured_layout == parameter_layout
    assert tuple(changed.resolved_metadata.sources) == (name,)
    assert changed.resolved_metadata.sources[name].dtype == expected_schema[0][1]
    assert changed.resolved_metadata.sources[name].global_shape == shape
    if name != "weight" or shape != (4,):
        plan = _plan_staged_transfer(
            capture, changed.resolved_metadata.sources
        )
        with pytest.raises(IncompleteRefit):
            _NixlStagedTransfer._validate_complete(capture, parameter_layout, plan)


def test_metadata_reuse_requires_matching_manifests(monkeypatch) -> None:
    manifests = [
        _manifest(agent_name="a", endpoint="a:19000", offset=0, address=100),
        _manifest(agent_name="b", endpoint="b:19000", offset=2, address=200),
    ]
    transfer, first = _transfer_with_resolved_plan(monkeypatch, manifests)
    metrics = {}
    copied = [bytes(bytearray(blob)) for blob in manifests]
    resolved = transfer._resolve_metadata(copied, metrics)
    assert resolved.sources == first.sources
    assert metrics["source_cache_hits"] == 1
    assert metrics["source_manifest_bytes"] == sum(map(len, manifests))
    for changed, expected_shards in (
        (list(reversed(manifests)), [(200, (2,)), (100, (0,))]),
        (manifests[:1], [(100, (0,))]),
    ):
        resolved = transfer._resolve_metadata(changed, metrics)
        assert [
            (shard.addr, shard.shard_offset)
            for shard in resolved.sources["weight"].shards
        ] == expected_shards
        assert metrics["source_cache_hits"] == 0


@pytest.mark.parametrize(
    "field", ["digest", "device_id", "agent_meta_b64", "publisher_step", "addr"]
)
def test_metadata_refreshes_changed_version_and_transport_fields(
    monkeypatch, field
) -> None:
    manifest = _manifest(agent_name="a", endpoint="a:19000", offset=0, address=100)
    transfer, first = _transfer_with_resolved_plan(monkeypatch, [manifest])
    payload = json.loads(manifest)
    if field in ("digest", "device_id", "addr"):
        payload["tensors"][0]["shards"][0][field] = {
            "digest": "new-digest",
            "device_id": 1,
            "addr": 300,
        }[field]
    else:
        payload[field] = "bmV3LW1ldGFkYXRh" if field == "agent_meta_b64" else 2
    changed = json.dumps(payload).encode()
    metrics = {}
    resolved = transfer._resolve_metadata([changed], metrics)
    expected = _resolve_sources([changed])
    assert resolved.session_to_agent == expected.session_to_agent
    assert resolved.session_to_device == expected.session_to_device
    assert resolved.agent_metadata == expected.agent_metadata
    for name, source in resolved.sources.items():
        original = expected.sources[name]
        assert (source.global_shape, source.dtype, source.elsize) == (
            original.global_shape,
            original.dtype,
            original.elsize,
        )
        assert [tuple(shard) for shard in source.shards] == [
            tuple(vars(shard).values()) for shard in original.shards
        ]
    assert first.sources["weight"].shards[0].addr == 100
    assert metrics["source_cache_misses"] == 1


@pytest.mark.parametrize(
    "defect", ["empty", "malformed", "duplicate", "mutable", "geometry"]
)
def test_metadata_rejects_bad_manifests_even_with_a_cached_plan(
    monkeypatch, defect
) -> None:
    manifest = _manifest(agent_name="a", endpoint="a:19000", offset=0, address=100)
    transfer, _ = _transfer_with_resolved_plan(monkeypatch, [manifest])
    if defect == "geometry":
        payload = json.loads(
            _manifest(agent_name="b", endpoint="b:19000", offset=2, address=200)
        )
        payload["tensors"][0]["full_shape"] = [8]
        bad = [manifest, json.dumps(payload).encode()]
    else:
        bad = {
            "empty": [],
            "malformed": [b"not-json"],
            "duplicate": [manifest, manifest],
            "mutable": [bytearray(manifest)],
        }[defect]
    with pytest.raises((ValueError, TypeError)):
        transfer._resolve_metadata(bad, {})


def test_warm_plan_transfers_changed_values_without_rebuilding(harness) -> None:
    first = harness.prepare()
    harness.collect(first)
    for tensor in harness.sources.values():
        tensor.add_(7)
    prepared = harness.prepare()
    metrics, installed = harness.collect(prepared)
    _check_values(harness, installed)
    assert prepared.metrics["plan_cache_hits"] == 1
    assert prepared.metrics.get("owner_plan_builds", 0) == 0
    assert prepared.metrics.get("source_decode_s", 0) == 0


@pytest.mark.parametrize(
    "defect", ["unattributed", "unsupported", "missing", "unknown", "fallback"]
)
def test_supplied_complete_plan_keeps_global_coverage_gate(monkeypatch, defect) -> None:
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


@pytest.mark.parametrize("diagnostic", ["plan", "layout"])
def test_diagnostic_validation_keeps_values_and_detects_layout_drift(harness, diagnostic) -> None:
    prepare = harness.prepare
    first = prepare()
    harness.collect(first)
    if diagnostic == "layout":
        changed_capture = CaptureResult(copies=[replace(copy) for copy in harness.capture.copies])
        changed_capture.copies[0] = replace(changed_capture.copies[0], dest_offset=1)
        snapshot = harness.transfer._weight_update_plan.trainer_source_snapshot
        reads = harness.events.count("post")
        with pytest.raises(RuntimeError, match="layout changed"):
            harness.transfer.prepare_streaming(
                trainer_snapshot=snapshot,
                manifests=[shard.metadata for shard in snapshot.shards],
                capture_layout=lambda _: (changed_capture, dict(harness.layout)),
            )
        assert harness.events.count("post") == reads
    else:
        prepared = prepare()
        _, installed = harness.collect(prepared)
        _check_values(harness, installed)
        assert prepared.metrics["bounded_whole_validation_s"] > 0

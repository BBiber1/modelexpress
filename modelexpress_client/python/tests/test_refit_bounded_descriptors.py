# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import hashlib
import gc
import weakref
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace, MappingProxyType

import modelexpress_rl.inference.nixl_staged_transfer as module
from modelexpress_rl.inference.adapter import TrainerSourceShard
from modelexpress_rl.inference.plan import StreamingSettings

import pytest
import torch
from modelexpress import envs
from modelexpress.accelerators import NIXL_ACCELERATOR_MEM_TYPE
from modelexpress.refit.reshard.rendezvous import (
    PublishedShard,
    PublishedTensor,
    wrap_rendezvous_blob,
)
from modelexpress.refit.reshard.verify import tensor_digest
from modelexpress_rl.inference.plan import TrainerSourceSnapshot
from modelexpress.refit.reshard.types import (
    CaptureResult,
    IncompleteRefit,
    RecordedCopy,
)


@pytest.fixture
def harness(monkeypatch, request) -> Iterator[SimpleNamespace]:
    """Real planning and byte copies with synthetic capture metadata and CPU arenas."""
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    monkeypatch.setenv("MX_RESHARD_MAX_SEGMENTS_PER_COPY", "1")
    parameters = getattr(getattr(request.node, "callspec", None), "params", {})
    settings = getattr(request, "param", StreamingSettings(1024, "cpu"))
    pack = parameters.get("pack", False)
    if isinstance(settings, tuple):
        settings, pack = settings
    monkeypatch.setenv("MX_REFIT_PACK_MODULES", str(int(pack)))
    monkeypatch.setenv("MX_REFIT_DEBUG_VALIDATE_WORKSPACE", str(int(parameters.get("workspace_debug", False))))
    diagnostic = parameters.get("diagnostic")
    monkeypatch.setenv("MX_REFIT_DEBUG_VALIDATE_PLAN", str(int(diagnostic == "plan")))
    monkeypatch.setenv("MX_REFIT_DEBUG_VALIDATE_GENERATOR_LAYOUT", str(int(diagnostic == "layout")))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    sources = {
        name: torch.arange(16, dtype=torch.float32).reshape(4, 4) + index / 16
        for index, name in enumerate(("exact", "full", "convert"))
    }
    captures = []
    copies = [
        RecordedCopy(src, ops, name, 2, (4, 4), (4, 1), dtype)
        for src, ops, name, dtype in (
            ("exact", (), "a.weight", torch.float32),
            ("full", (("transpose", (0, 1), ()),), "b.weight", torch.float32),
            ("convert", (), "c.weight", torch.bfloat16),
            ("exact", (), "d.weight", torch.float32),
        )
    ]
    capture = CaptureResult(copies=copies)
    layout = {copy.param_name: ((20,), copy.dest_dtype) for copy in copies}
    transports = []
    events = []

    class Manager:
        def __init__(self, **kwargs) -> None:
            self.registered = {}

        def initialize(self) -> None:
            events.append("initialize")

        def shutdown(self) -> None:
            events.append("shutdown")
            self.registered.clear()

        def add_remote_agent(self, metadata) -> str:
            events.append(f"connect:{metadata.decode()}")
            return metadata.decode().split(":", 1)[0]

        def register_tensors(self, tensors) -> None:
            events.append("register")
            self.registered.update(tensors)

        def register_dram_buffer(self, tensor) -> object:
            events.append("register")
            handle = object()
            self.registered[handle] = tensor
            return handle

        def deregister_memory(self, handle) -> None:
            events.append("deregister")
            del self.registered[handle]

    class Transport:
        def __init__(self, manager, agents, devices, **kwargs) -> None:
            self.posts = []
            self.posted = []
            self.awaited = []
            self.fail_post = False
            self.fail_wait = False
            # An omitted override uses the manager's accelerator memory type.
            self.mem_type = kwargs["local_mem_type"] or NIXL_ACCELERATOR_MEM_TYPE
            transports.append(self)

        def post_reads(self, descriptors) -> list[SimpleNamespace]:
            events.append("post")
            if self.fail_post:
                raise RuntimeError("post failed")
            self.posts.append(tuple(descriptors))
            for descriptor in descriptors:
                ctypes.memmove(
                    descriptor.dst_addr, descriptor.src_addr, descriptor.nbytes
                )
            handle = SimpleNamespace()
            self.posted.append(handle)
            # The transport owns its argument list; changing it must not alter cache.
            descriptors.clear()
            return [handle]

        def await_reads(self, posted) -> None:
            events.append("await")
            if self.fail_wait:
                raise RuntimeError("wait failed")
            self.awaited.extend(posted)

    monkeypatch.setattr(module, "NixlTransferManager", Manager)
    monkeypatch.setattr(module, "NixlReshardTransport", Transport)
    monkeypatch.setattr(
        module._NixlStagedTransfer,
        "_allocate_arena",
        lambda self, size: torch.empty(size, dtype=torch.uint8),
    )
    transfer = module._NixlStagedTransfer(
        agent_name="target",
        device_id=0,
        device=torch.device("cuda:0"),
        listen_port=None,
        streaming=settings,
    )

    def manifests() -> list[bytes]:
        return [
            wrap_rendezvous_blob(
                b"source",
                "source",
                "source:19000",
                [
                    PublishedTensor(
                        name=name,
                        dtype="torch.float32",
                        elsize=4,
                        full_shape=(4, 4),
                        shards=[
                            PublishedShard(
                                agent_name="source",
                                device_id=0,
                                addr=tensor.data_ptr(),
                                shard_offset=(0, 0),
                                shape=(4, 4),
                            )
                        ],
                    )
                    for name, tensor in sources.items()
                ],
            )
        ]

    def capture_layout(manifest) -> tuple[CaptureResult, dict]:
        captures.append(manifest)
        return capture, layout

    def prepare(*, worker="worker", generation=1) -> module._PreparedBoundedTransfer:
        blobs = manifests()
        snapshot = TrainerSourceSnapshot("mesh", generation, tuple(
            TrainerSourceShard("slot", worker, hashlib.sha256(blob).hexdigest(), "source:19000", blob)
            for blob in blobs
        ))
        cached = transfer.cached_trainer_source()
        if cached is not None and cached.physical_fingerprint == snapshot.physical_fingerprint:
            snapshot = cached
        return transfer.prepare_streaming(
            trainer_snapshot=snapshot,
            manifests=blobs,
            capture_layout=capture_layout,
            checksums=MappingProxyType({
                (name, "source", tensor.data_ptr(), (0, 0), (4, 4)): tensor_digest(tensor)
                for name, tensor in sources.items()
            }) if envs.MX_RESHARD_PUBLISH_DIGEST else None,
        )

    def collect(prepared) -> tuple[dict, dict]:
        metrics, installed = {}, {}
        for tensors in transfer.iter_bounded(prepared, metrics):
            installed.update({name: value.clone() for name, value in tensors.items()})
        return metrics, installed

    state = SimpleNamespace(
        transfer=transfer,
        streaming=settings,
        sources=sources,
        capture=capture,
        layout=layout,
        prepare=prepare,
        collect=collect,
        captures=captures,
        transports=transports,
        events=events,
    )
    yield state
    transfer.close()


def _check_values(harness, installed, *, exact_rows: int | None = None, check_padding: bool = True) -> None:
    for source, name, dtype, transpose in (
        ("exact", "a.weight", torch.float32, False),
        ("full", "b.weight", torch.float32, True),
        ("convert", "c.weight", torch.bfloat16, False),
        ("exact", "d.weight", torch.float32, False),
    ):
        expected = torch.zeros(20, dtype=dtype)
        value = harness.sources[source]
        if source == "exact" and exact_rows is not None:
            value = value[:exact_rows]
        expected[2:18].copy_((value.T if transpose else value).reshape(-1))
        if check_padding:
            assert torch.equal(installed[name], expected)
        else:
            assert torch.equal(installed[name][2:18], expected[2:18])


@pytest.mark.parametrize(
    "harness",
    [
        StreamingSettings(1024, device, count)
        for device in ("cpu", "cuda")
        for count in (1, 2)
    ],
    indirect=True,
)
@pytest.mark.parametrize("pack", [False, True])
def test_warm_descriptors_still_transfer_new_values(harness, monkeypatch, pack) -> None:
    monkeypatch.setenv("MX_REFIT_PACK_MODULES", str(int(pack)))
    prepare = harness.prepare
    first = prepare()
    assert any(batch.transfer_plan.full_pulls for batch in first.batches)
    assert any(batch.transfer_plan.converts for batch in first.batches)
    assert any(batch.transfer_plan.segments for batch in first.batches)
    cold, installed = harness.collect(first)
    _check_values(harness, installed)
    assert cold["descriptor_builds"] == len(first.batches)
    assert cold["descriptor_cache_hits"] == 0
    for values in harness.sources.values():
        values.add_(3)
    for arena in harness.transfer._staging_arenas:
        arena.fill_(255)
    second = prepare()
    warm, installed = harness.collect(second)
    _check_values(harness, installed)
    assert warm["descriptor_cache_hits"] == len(second.batches)
    assert warm["descriptor_cache_misses"] == warm["descriptor_builds"] == 0
    assert second.transport.mem_type == (
        "DRAM" if harness.streaming.staging_device == "cpu" else "VRAM"
    )
    assert first.transport.posts == second.transport.posts
    assert second.transport.awaited == second.transport.posted
    descriptor = second.transport.posts[0][0]
    with pytest.raises(AttributeError):
        descriptor.dst_addr = 0


def test_changed_source_addresses_rebuild_descriptors(harness) -> None:
    first = harness.prepare()
    harness.collect(first)
    harness.sources["exact"] = harness.sources["exact"].clone() + 5
    second = harness.prepare(generation=2)
    metrics, _ = harness.collect(second)
    assert metrics["descriptor_cache_hits"] == 0
    assert metrics["descriptor_builds"] == len(second.batches)


@pytest.mark.parametrize("workspace_debug", [True])
@pytest.mark.parametrize(
    "change", ["replace", "same_address", "reorder", "resize", "registration"]
)
@pytest.mark.parametrize("harness", [StreamingSettings(1024, "cpu", 2)], indirect=True)
def test_arena_change_after_prepare_is_checked_before_each_post(
    harness, change, workspace_debug
) -> None:
    first = harness.prepare()
    harness.collect(first)
    prepared = harness.prepare()
    arenas = harness.transfer._staging_arenas
    if change == "replace":
        arenas[1] = arenas[1].clone()
    elif change == "same_address":
        arenas[1] = arenas[1].view_as(arenas[1])
    elif change == "reorder":
        arenas.reverse()
    elif change == "resize":
        arenas[1].resize_(arenas[1].numel() + 256)
    else:
        harness.transfer._release_staging_registrations()
        harness.transfer._staging_registrations = [
            harness.transfer._manager.register_dram_buffer(arena) for arena in arenas
        ]
    with pytest.raises(RuntimeError, match="bounded workspace changed"):
        harness.collect(prepared)
    assert prepared.transport.posts == []


@pytest.mark.parametrize("workspace_debug", [True])
def test_arena_change_between_batches_is_not_hidden_by_first_hit(harness, workspace_debug) -> None:
    harness.collect(harness.prepare())
    prepared = harness.prepare()
    metrics = {}
    iterator = harness.transfer.iter_bounded(prepared, metrics)
    next(iterator)
    assert metrics["descriptor_cache_hits"] == 1
    harness.transfer._staging_arenas[0] = harness.transfer._staging_arenas[0].clone()
    posts_before_change = len(prepared.transport.posts)
    with pytest.raises(RuntimeError, match="bounded workspace changed"):
        next(iterator)
    assert len(prepared.transport.posts) == posts_before_change
    assert prepared.transport.awaited == prepared.transport.posted


@pytest.mark.parametrize(
    "failure", ["coverage", "transport", "prepared", "registration"]
)
def test_failed_prepare_discards_descriptors(harness, monkeypatch, failure) -> None:
    harness.collect(harness.prepare())
    with monkeypatch.context() as fault:
        if failure == "coverage":
            harness.sources["exact"] = harness.sources["exact"].clone()
            harness.layout["missing.weight"] = ((4,), torch.float32)
            expected = IncompleteRefit
        else:
            def fail(*args, **kwargs) -> None:
                raise RuntimeError("preparation failed")

            if failure == "transport":
                fault.setattr(module, "NixlReshardTransport", fail)
            elif failure == "prepared":
                fault.setattr(module, "_PreparedBoundedTransfer", fail)
            else:
                harness.transfer.reset_workspace()
                fault.setattr(harness.transfer._manager, "register_dram_buffer", fail)
            expected = RuntimeError
        with pytest.raises(expected):
            harness.prepare(generation=2 if failure == "coverage" else 1)
    harness.layout.pop("missing.weight", None)
    prepared = harness.prepare(generation=2 if failure == "coverage" else 1)
    metrics, installed = harness.collect(prepared)
    _check_values(harness, installed)
    assert metrics["descriptor_builds"] == len(prepared.batches)
    assert prepared.transport.awaited == prepared.transport.posted


@pytest.mark.parametrize("failure", ["post", "wait", "abandon", "drain"])
@pytest.mark.parametrize("harness", [StreamingSettings(1024, "cpu", 2)], indirect=True)
def test_incomplete_iteration_discards_cache_and_preserves_drain(
    harness, failure
) -> None:
    harness.collect(harness.prepare())
    prepared = harness.prepare()
    if failure in ("post", "wait"):
        setattr(prepared.transport, f"fail_{failure}", True)
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            harness.collect(prepared)
    else:
        iterator = harness.transfer.iter_bounded(prepared, {})
        next(iterator)
        assert len(prepared.transport.posted) == 2
        if failure == "drain":
            prepared.transport.fail_wait = True
            with pytest.raises(RuntimeError, match="could not be drained"):
                iterator.close()
            assert len(prepared.transport.awaited) < len(prepared.transport.posted)
            registered = harness.transfer._manager.registered.values()
            assert all(
                any(resource is arena for resource in registered)
                for arena in harness.transfer._staging_arenas
            )
            return
        else:
            iterator.close()
            assert prepared.transport.awaited == prepared.transport.posted
    prepared = harness.prepare()
    metrics, installed = harness.collect(prepared)
    _check_values(harness, installed)
    assert metrics["descriptor_builds"] == len(prepared.batches)
    assert prepared.transport.awaited == prepared.transport.posted


@pytest.mark.parametrize("cleanup", ["reset", "close"])
def test_reset_and_close_release_arenas_and_handles(harness, cleanup) -> None:
    prepared = harness.prepare()
    harness.collect(prepared)
    arena_refs = [weakref.ref(arena) for arena in harness.transfer._staging_arenas]
    transport_ref = weakref.ref(prepared.transport)
    # Drop test-only observation lists and all prepared state.
    harness.transports.clear()
    del prepared
    getattr(harness.transfer, cleanup if cleanup == "close" else "reset_workspace")()
    gc.collect()
    assert all(reference() is None for reference in arena_refs)
    assert transport_ref() is None


def _larger_source_snapshot(harness) -> tuple[TrainerSourceSnapshot, list[torch.Tensor]]:
    source = harness.sources["exact"]
    rows = [source[start:start + 2].clone() for start in range(0, source.shape[0], 2)]
    blob = wrap_rendezvous_blob(
        b"source", "source", "source:19000",
        [
            PublishedTensor(
                name=name, dtype="torch.float32", elsize=4,
                full_shape=tuple(source.shape) if name == "exact" else (4, 4),
                shards=[
                    PublishedShard(
                        agent_name="source", device_id=0, addr=value.data_ptr(),
                        shard_offset=(2 * index, 0), shape=(2, 4),
                    ) for index, value in enumerate(rows)
                ] if name == "exact" else [PublishedShard(
                    agent_name="source", device_id=0, addr=tensor.data_ptr(),
                    shard_offset=(0, 0), shape=(4, 4),
                )],
            ) for name, tensor in harness.sources.items()
        ],
    )
    snapshot = TrainerSourceSnapshot("mesh", 2, (TrainerSourceShard(
        "slot", "worker", hashlib.sha256(blob).hexdigest(), "source:19000", blob,
    ),))
    return snapshot, rows


@pytest.mark.parametrize("harness", [(StreamingSettings(2048, "cpu"), True)], indirect=True)
def test_descriptor_order_duplicates_and_empty_reads_survive_reuse(
    harness, monkeypatch
) -> None:
    monkeypatch.setenv("MX_REFIT_PACK_MODULES", "1")
    original = harness.transfer._descriptors
    builds = []

    def descriptors(*args, **kwargs):
        result = original(*args, **kwargs)
        result += [result[0], replace(result[0], nbytes=0)]
        builds.append(
            tuple((d.session, d.src_addr, d.dst_addr, d.nbytes) for d in result)
        )
        return result

    monkeypatch.setattr(harness.transfer, "_descriptors", descriptors)
    first = harness.prepare()
    assert len(first.batches) == 1
    harness.collect(first)
    second = harness.prepare()
    metrics, installed = harness.collect(second)
    _check_values(harness, installed)
    assert tuple(second.transport.posts[0]) == builds[0]
    assert metrics["descriptor_cache_hits"] == 1


def test_descriptor_build_time_stays_outside_wire_time(harness, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: now[0])
    original = harness.transfer._descriptors

    def delayed_descriptors(*args, **kwargs):
        now[0] += 100
        return original(*args, **kwargs)

    monkeypatch.setattr(harness.transfer, "_descriptors", delayed_descriptors)
    cold, _ = harness.collect(harness.prepare())
    warm, _ = harness.collect(harness.prepare())
    assert now[0] == 400
    assert cold["wire_s"] == warm["wire_s"] == 0
    assert warm["descriptor_builds"] == 0


@pytest.mark.parametrize("tensor_indices", [False, True])
def test_prepared_list_indexing_replays_selected_rows_with_fresh_values(
    harness, monkeypatch, tensor_indices
) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")
    indices = torch.tensor([0, 2]) if tensor_indices else [0, 2]
    copy = harness.capture.copies[1]
    copy.op_chain = (("__getitem__", (indices,), ()),)
    copy.dest_shape = (2, 4)
    copy.dest_stride = (4, 1)
    prepared = harness.prepare()
    _, installed = harness.collect(prepared)
    expected = torch.zeros(20, dtype=torch.float32)
    expected[2:10].copy_(harness.sources["full"][[0, 2]].reshape(-1))
    assert torch.equal(installed["b.weight"], expected)
    _, installed = harness.collect(harness.prepare())
    assert torch.equal(installed["b.weight"], expected)
    harness.sources["full"].add_(5)
    _, installed = harness.collect(harness.prepare())
    expected[2:10].copy_(harness.sources["full"][[0, 2]].reshape(-1))
    assert torch.equal(installed["b.weight"], expected)


@pytest.mark.parametrize("aliases", [False, True])
def test_compatible_replicas_extend_plan_and_read_current_bytes(harness, aliases) -> None:
    if aliases:
        harness.sources.update({name: harness.sources["exact"] for name in harness.sources})
    replicas = {"A": dict(harness.sources)}
    replicas["B"] = {name: tensor.clone() + 100 + index for index, (name, tensor) in enumerate(reversed(tuple(harness.sources.items())))}
    for index, worker in enumerate(("A", "B", "A")):
        harness.sources.update(replicas[worker])
        prepared = harness.prepare(worker=worker)
        _, installed = harness.collect(prepared)
        _check_values(harness, installed)
        if index:
            assert prepared.metrics.get("initial_whole_plan_s", 0) == 0
            assert prepared.metrics.get("owner_plan_builds", 0) == 0
            assert prepared.metrics["replica_rebindings"] == 1


def test_known_replica_storage_drift_fails_before_read(harness) -> None:
    harness.collect(harness.prepare(worker="A"))
    harness.sources["exact"] = harness.sources["exact"].clone() + 100
    harness.events.clear()
    with pytest.raises(RuntimeError, match="within a mesh generation"):
        harness.prepare(worker="A")
    assert "post" not in harness.events
    prepared = harness.prepare(worker="A", generation=2)
    _, installed = harness.collect(prepared)
    _check_values(harness, installed)


@pytest.mark.parametrize("conflicting_agent", [False, True])
def test_replica_binding_preserves_original_slot_owner(harness, conflicting_agent) -> None:
    def shard(slot: str, worker: str, agent: str, values: dict) -> TrainerSourceShard:
        blob = wrap_rendezvous_blob(
            agent.encode(), agent, f"{agent}:19000",
            [PublishedTensor(
                name=name, dtype="torch.float32", elsize=4, full_shape=(4, 4),
                shards=[PublishedShard(agent_name=agent, device_id=0, addr=tensor.data_ptr(), shard_offset=(0, 0), shape=(4, 4))],
            ) for name, tensor in values.items()],
        )
        return TrainerSourceShard(slot, worker, hashlib.sha256(blob).hexdigest(), f"{agent}:19000", blob)

    def prepare(shards: tuple) -> module._PreparedBoundedTransfer:
        snapshot = TrainerSourceSnapshot("owners", 1, shards)
        cached = harness.transfer.cached_trainer_source()
        if cached is not None and cached.physical_fingerprint == snapshot.physical_fingerprint:
            snapshot = cached
        return harness.transfer.prepare_streaming(
            trainer_snapshot=snapshot,
            manifests=[source.metadata for source in shards],
            capture_layout=lambda _: (harness.capture, harness.layout),
        )

    original = dict(harness.sources)
    ignored = {name: tensor.clone() + 1000 for name, tensor in original.items()}
    replacement = {name: tensor.clone() + 100 for name, tensor in original.items()}
    a = shard("first", "A", "agentA", original)
    z = shard("second", "Z", "agentZ", ignored)
    _, installed = harness.collect(prepare((a, z)))
    _check_values(harness, installed)
    assert "connect:agentA" in harness.events
    assert "connect:agentZ" not in harness.events
    b = shard("first", "B", "agentZ" if conflicting_agent else "agentB", replacement)
    harness.events.clear()
    if conflicting_agent:
        with pytest.raises(ValueError, match="duplicate NIXL agents"):
            prepare((z, b))
        assert "post" not in harness.events
    else:
        harness.sources.update(replacement)
        prepared = prepare((z, b))
        _, installed = harness.collect(prepared)
        _check_values(harness, installed)
        assert prepared.metrics.get("owner_plan_builds", 0) == 0
        assert "connect:agentB" in harness.events
        assert "connect:agentZ" not in harness.events
        harness.events.clear()
        for tensor in replacement.values():
            tensor.add_(1)
        _, installed = harness.collect(prepare((z, b)))
        _check_values(harness, installed)
        assert not any(event.startswith("connect:") for event in harness.events)
        harness.transfer.reset_workspace()
        prepared = prepare((b, z))
        _, installed = harness.collect(prepared)
        _check_values(harness, installed)
        assert prepared.metrics["plan_cache_misses"] == 1
        assert "connect:agentB" in harness.events
        assert "connect:agentZ" not in harness.events


def test_failed_replica_setup_rebuilds_before_next_transfer(harness, monkeypatch) -> None:
    original = dict(harness.sources)
    harness.collect(harness.prepare(worker="A"))
    harness.sources.update({name: tensor.clone() + 100 for name, tensor in original.items()})
    transport = module.NixlReshardTransport

    def fail(*args, **kwargs) -> None:
        raise RuntimeError("replica connection failed")

    monkeypatch.setattr(module, "NixlReshardTransport", fail)
    with pytest.raises(RuntimeError, match="replica connection failed"):
        harness.prepare(worker="B")
    monkeypatch.setattr(module, "NixlReshardTransport", transport)
    harness.sources.update(original)
    prepared = harness.prepare(worker="A")
    _, installed = harness.collect(prepared)
    _check_values(harness, installed)
    assert prepared.metrics["plan_cache_misses"] == 1
    assert prepared.metrics["initial_whole_plan_s"] > 0


def test_registration_generation_drift_rejects_without_debug_validation(harness) -> None:
    prepared = harness.prepare()
    harness.transfer._release_staging_registrations()
    with pytest.raises(RuntimeError, match="bounded workspace changed"):
        harness.collect(prepared)
    assert prepared.transport.posts == []


def test_partial_iteration_reports_all_eager_descriptor_builds(harness) -> None:
    prepared = harness.prepare()
    assert prepared.metrics["descriptor_builds"] == len(prepared.batches)
    metrics = {}
    iterator = harness.transfer.iter_bounded(prepared, metrics)
    next(iterator)
    assert metrics["descriptor_builds"] == len(prepared.batches)
    assert len(prepared.transport.posts) < len(prepared.batches)
    iterator.close()
    assert prepared.transport.awaited == prepared.transport.posted


@pytest.mark.parametrize("harness", [(StreamingSettings(2048, "cpu", buffers), buffers == 2) for buffers in (1, 2)], indirect=True)
def test_larger_source_scratch_grows_then_reuses_capacity(harness) -> None:
    if harness.streaming.staging_buffers == 1:
        exact = harness.sources["exact"]
        harness.sources["exact"] = torch.cat((exact, torch.zeros((16, 4), dtype=exact.dtype)))
        for copy in harness.capture.copies:
            if copy.src_name == "exact":
                copy.op_chain = (("__getitem__", (slice(0, 4),), ()),)
    initial = harness.prepare()
    cold, installed = harness.collect(initial)
    _check_values(harness, installed, exact_rows=4)
    for tensor in harness.sources.values():
        tensor.add_(5)
    snapshot, _rows = _larger_source_snapshot(harness)
    blob = snapshot.shards[0].metadata
    larger = harness.transfer.prepare_streaming(
        manifests=[blob], trainer_snapshot=snapshot,
        capture_layout=lambda _: (harness.capture, harness.layout),
    )
    grown, installed = harness.collect(larger)
    _check_values(harness, installed, exact_rows=4)
    assert grown["staging_peak_bytes"] > cold["staging_peak_bytes"]
    assert grown["staging_peak_bytes"] <= harness.streaming.max_staging_bytes
    current_addresses = {arena.data_ptr() for arena in harness.transfer._staging_arenas}
    assert harness.events.count("connect:source") == 2
    registered = list(harness.transfer._manager.registered.values())
    assert sum(value.numel() for value in registered) == grown["staging_peak_bytes"]
    for descriptors in larger.transport.posts:
        assert all(any(
            value.data_ptr() <= descriptor.dst_addr
            and descriptor.dst_addr + descriptor.nbytes <= value.data_ptr() + value.numel()
            for value in registered
        ) for descriptor in descriptors)
    transition = harness.events[harness.events.index("shutdown"):]
    assert transition.index("shutdown") < transition.index("initialize") < transition.index("connect:source") < transition.index("register") < transition.index("post")
    for tensor in harness.sources.values():
        tensor.add_(2)
    smaller = harness.prepare(generation=3)
    retained, installed = harness.collect(smaller)
    _check_values(harness, installed, exact_rows=4)
    assert retained["staging_peak_bytes"] == grown["staging_peak_bytes"]
    assert {arena.data_ptr() for arena in harness.transfer._staging_arenas} == current_addresses
    assert harness.events.count("connect:source") == 2
    assert larger.transport.awaited == larger.transport.posted
    assert smaller.transport.awaited == smaller.transport.posted


@pytest.mark.parametrize("harness", [(StreamingSettings(2048, "cpu", 2), True)], indirect=True)
def test_borrowed_manager_rejects_growth_without_native_teardown(harness, monkeypatch) -> None:
    harness.collect(harness.prepare())
    snapshot, _rows = _larger_source_snapshot(harness)
    registrations = [(value.data_ptr(), value.numel()) for value in harness.transfer._manager.registered.values()]
    harness.events.clear()
    with monkeypatch.context() as borrowed:
        borrowed.setattr(harness.transfer, "_owns_manager", False)
        with pytest.raises(RuntimeError, match="transfer-owned NIXL agent"):
            harness.transfer.prepare_streaming(
                manifests=[snapshot.shards[0].metadata], trainer_snapshot=snapshot,
                capture_layout=lambda _: (harness.capture, harness.layout),
            )
    assert not any(event in harness.events for event in ("shutdown", "post", "register", "deregister"))
    assert [(value.data_ptr(), value.numel()) for value in harness.transfer._manager.registered.values()] == registrations


@pytest.mark.parametrize("harness", [(StreamingSettings(2048, "cpu", 2), True)], indirect=True)
def test_owned_growth_registration_failure_cleans_up_before_retry(harness, monkeypatch) -> None:
    harness.collect(harness.prepare())
    snapshot, _rows = _larger_source_snapshot(harness)
    harness.events.clear()

    def fail_registration(_tensor: torch.Tensor) -> None:
        harness.events.append("register_failed")
        raise RuntimeError("growth registration failed")

    with monkeypatch.context() as fault:
        fault.setattr(harness.transfer._manager, "register_dram_buffer", fail_registration)
        with pytest.raises(RuntimeError, match="growth registration failed"):
            harness.transfer.prepare_streaming(
                manifests=[snapshot.shards[0].metadata], trainer_snapshot=snapshot,
                capture_layout=lambda _: (harness.capture, harness.layout),
            )
    assert "post" not in harness.events
    assert harness.transfer._manager.registered == {}
    assert harness.events.index("shutdown") < harness.events.index("initialize") < harness.events.index("connect:source") < harness.events.index("register_failed")
    assert harness.events[-1] == "shutdown"
    recovered = harness.transfer.prepare_streaming(
        manifests=[snapshot.shards[0].metadata], trainer_snapshot=snapshot,
        capture_layout=lambda _: (harness.capture, harness.layout),
    )
    _, installed = harness.collect(recovered)
    _check_values(harness, installed)
    assert recovered.transport.awaited == recovered.transport.posted


def _agent_registration_snapshot(
    harness, *, generation: int, tokens: dict[str, bytes],
    assignments: dict[str, str] | None = None, worker: str = "worker",
) -> TrainerSourceSnapshot:
    assignments = assignments or {name: "source" for name in harness.sources}
    shards = []
    for agent, token in tokens.items():
        blob = wrap_rendezvous_blob(
            token, agent, f"{agent}:19000",
            [PublishedTensor(
                name=name, dtype="torch.float32", elsize=4, full_shape=(4, 4),
                shards=[PublishedShard(
                    agent_name=agent, device_id=0, addr=tensor.data_ptr(),
                    shard_offset=(0, 0), shape=(4, 4),
                )],
            ) for name, tensor in harness.sources.items() if assignments[name] == agent],
        )
        shards.append(TrainerSourceShard(agent, worker, hashlib.sha256(blob).hexdigest(), f"{agent}:19000", blob))
    return TrainerSourceSnapshot("registration-mesh", generation, tuple(shards))


def _prepare_agent_registration(harness, snapshot) -> module._PreparedNixlTransfer | module._PreparedBoundedTransfer:
    prepare = harness.transfer.prepare_full_copy if harness.streaming is None else harness.transfer.prepare_streaming
    return prepare(
        manifests=[shard.metadata for shard in snapshot.shards], trainer_snapshot=snapshot,
        capture_layout=lambda _: (harness.capture, harness.layout),
    )


def _collect_agent_registration(harness, prepared) -> dict:
    if harness.streaming is not None:
        _, installed = harness.collect(prepared)
        return installed
    return {name: value.clone() for name, value in harness.transfer.stage(prepared).tensors.items()}


@pytest.mark.parametrize("harness", [None, StreamingSettings(1024, "cpu")], indirect=True)
def test_next_generation_replaces_agent_registration_before_read(harness, monkeypatch) -> None:
    monkeypatch.setattr(harness.transfer, "_device", torch.device("cpu"))
    monkeypatch.setattr(module, "classic_cuda_alloc", nullcontext)
    first = _agent_registration_snapshot(harness, generation=1, tokens={"source": b"source"})
    _check_values(harness, _collect_agent_registration(harness, _prepare_agent_registration(harness, first)), check_padding=harness.streaming is not None)
    for tensor in harness.sources.values():
        tensor.add_(5)
    harness.events.clear()
    second = _agent_registration_snapshot(harness, generation=2, tokens={"source": b"source:registration-v2"})
    prepared = _prepare_agent_registration(harness, second)
    _check_values(harness, _collect_agent_registration(harness, prepared), check_padding=harness.streaming is not None)
    events = harness.events
    assert events.index("shutdown") < events.index("initialize") < events.index("connect:source:registration-v2") < events.index("register") < events.index("post")
    assert prepared.transport.awaited == prepared.transport.posted


@pytest.mark.parametrize("worker", ["worker", "replacement"])
def test_current_generation_registration_conflict_rejects_before_read(harness, worker) -> None:
    first = _agent_registration_snapshot(harness, generation=1, tokens={"source": b"source"})
    _collect_agent_registration(harness, _prepare_agent_registration(harness, first))
    harness.events.clear()
    changed = _agent_registration_snapshot(harness, generation=1, tokens={"source": b"source:registration-v2"}, worker=worker)
    with pytest.raises(RuntimeError):
        _prepare_agent_registration(harness, changed)
    assert "post" not in harness.events
    assert "connect:source:registration-v2" not in harness.events


def test_borrowed_manager_registration_replacement_preserves_native_resources(harness, monkeypatch) -> None:
    first = _agent_registration_snapshot(harness, generation=1, tokens={"source": b"source"})
    _collect_agent_registration(harness, _prepare_agent_registration(harness, first))
    registrations = [(tensor.data_ptr(), tensor.numel()) for tensor in harness.transfer._manager.registered.values()]
    harness.events.clear()
    second = _agent_registration_snapshot(harness, generation=2, tokens={"source": b"source:registration-v2"})
    with monkeypatch.context() as borrowed:
        borrowed.setattr(harness.transfer, "_owns_manager", False)
        with pytest.raises(RuntimeError, match="transfer-owned NIXL agent"):
            _prepare_agent_registration(harness, second)
    assert not any(event in harness.events for event in ("shutdown", "post", "deregister"))
    assert [(tensor.data_ptr(), tensor.numel()) for tensor in harness.transfer._manager.registered.values()] == registrations


@pytest.mark.parametrize("current_conflict", [False, True])
def test_required_agent_generations_distinguish_unused_and_current_conflicts(harness, current_conflict) -> None:
    split = {"exact": "source", "full": "other", "convert": "other"}
    first = _agent_registration_snapshot(harness, generation=1, tokens={"source": b"source", "other": b"other"}, assignments=split)
    _check_values(harness, _collect_agent_registration(harness, _prepare_agent_registration(harness, first)))
    second = _agent_registration_snapshot(harness, generation=2, tokens={"other": b"other"}, assignments={name: "other" for name in harness.sources})
    _check_values(harness, _collect_agent_registration(harness, _prepare_agent_registration(harness, second)))
    harness.events.clear()
    tokens = {"source": b"source:registration-v2", "other": b"other:registration-v2" if current_conflict else b"other"}
    third = _agent_registration_snapshot(harness, generation=2, tokens=tokens, assignments=split, worker="replacement")
    if current_conflict:
        with pytest.raises(RuntimeError, match="already connected source"):
            _prepare_agent_registration(harness, third)
        assert not any(event in harness.events for event in ("shutdown", "post", "initialize"))
    else:
        prepared = _prepare_agent_registration(harness, third)
        _check_values(harness, _collect_agent_registration(harness, prepared))
        assert harness.events.index("shutdown") < harness.events.index("connect:source:registration-v2") < harness.events.index("post")
        assert "connect:other" in harness.events


def test_registration_replacement_failure_cleans_up_and_retries_current_bytes(harness, monkeypatch) -> None:
    first = _agent_registration_snapshot(harness, generation=1, tokens={"source": b"source"})
    _collect_agent_registration(harness, _prepare_agent_registration(harness, first))
    for tensor in harness.sources.values():
        tensor.add_(5)
    second = _agent_registration_snapshot(harness, generation=2, tokens={"source": b"source:registration-v2"})
    harness.events.clear()

    def failed_connection(_metadata: bytes) -> None:
        harness.events.append("connect_failed")
        raise RuntimeError("registration replacement failed")

    with monkeypatch.context() as fault:
        fault.setattr(harness.transfer._manager, "add_remote_agent", failed_connection)
        with pytest.raises(RuntimeError, match="registration replacement failed"):
            _prepare_agent_registration(harness, second)
    assert "post" not in harness.events
    assert harness.transfer._manager.registered == {}
    assert harness.events.index("shutdown") < harness.events.index("initialize") < harness.events.index("connect_failed")
    assert harness.events[-1] == "shutdown"
    prepared = _prepare_agent_registration(harness, second)
    _check_values(harness, _collect_agent_registration(harness, prepared))
    assert prepared.transport.awaited == prepared.transport.posted

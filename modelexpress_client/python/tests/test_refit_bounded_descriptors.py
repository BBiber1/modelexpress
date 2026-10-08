# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import gc
import weakref
from collections.abc import Iterator
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import modelexpress_rl.inference.nixl_staged_transfer as module
import pytest
import torch
from modelexpress import envs
from modelexpress.accelerators import NIXL_ACCELERATOR_MEM_TYPE
from modelexpress.refit.reshard.rendezvous import (
    PublishedShard,
    PublishedTensor,
    wrap_rendezvous_blob,
)
from modelexpress.refit.reshard.types import (
    CaptureResult,
    IncompleteRefit,
    RecordedCopy,
)
from modelexpress.refit.reshard.verify import tensor_digest
from modelexpress_rl.inference.plan import TrainerSourceSnapshot


@pytest.fixture
def harness(monkeypatch) -> Iterator[SimpleNamespace]:
    """Real planning and byte copies with synthetic capture metadata and CPU arenas."""
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    monkeypatch.setenv("MX_RESHARD_MAX_SEGMENTS_PER_COPY", "1")
    monkeypatch.setenv("MX_REFIT_CACHE_PLAN", "1")
    monkeypatch.setenv("MX_REFIT_CACHE_GENERATOR_LAYOUT", "1")
    monkeypatch.setenv("MX_REFIT_PACK_MODULES", "0")
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
    remote = {"name": "source", "remove_ok": True, "fail_registration": None}

    class Manager:
        def __init__(self, **kwargs) -> None:
            self.registered = {}
            self.epoch = 0
            self.registration_calls = 0

        def initialize(self):
            pass

        def shutdown(self) -> None:
            self.epoch += 1
            self.registered.clear()

        def add_remote_agent(self, metadata):
            return metadata.decode()

        def remove_remote_agent(self, agent) -> bool:
            events.append("remove:" + agent)
            return remote["remove_ok"]

        def register_tensors(self, tensors) -> None:
            self.registered.update(tensors)
            self.registration_calls += 1
            if self.registration_calls == remote["fail_registration"]:
                raise RuntimeError("registration failed")

        def register_dram_buffer(self, tensor) -> object:
            handle = object()
            self.registered[handle] = tensor
            self.registration_calls += 1
            if self.registration_calls == remote["fail_registration"]:
                raise RuntimeError("registration failed")
            return handle

        def deregister_memory(self, handle):
            del self.registered[handle]

    class Transport:
        def __init__(self, manager, agents, devices, **kwargs) -> None:
            self.epoch = manager.epoch
            self.manager = manager
            self.agent_sessions = agents
            self.posts = []
            self.posted = []
            self.awaited = []
            self.fail_post = False
            self.fail_wait = False
            # An omitted override uses the manager's accelerator memory type.
            self.mem_type = kwargs["local_mem_type"] or NIXL_ACCELERATOR_MEM_TYPE
            transports.append(self)

        def post_reads(self, descriptors) -> list[SimpleNamespace]:
            assert self.epoch == self.manager.epoch
            events.append("post")
            if self.fail_post:
                raise RuntimeError("post failed")
            self.posts.append(tuple(descriptors))
            for descriptor in descriptors:
                assert descriptor.session in self.agent_sessions
                assert any(
                    tensor.data_ptr() <= descriptor.dst_addr
                    and descriptor.dst_addr + descriptor.nbytes
                    <= tensor.data_ptr() + tensor.numel() * tensor.element_size()
                    for tensor in self.manager.registered.values()
                )
                ctypes.memmove(
                    descriptor.dst_addr, descriptor.src_addr, descriptor.nbytes
                )
            handle = SimpleNamespace()
            self.posted.append(handle)
            # The transport owns its argument list; changing it must not alter cache.
            descriptors.clear()
            return [handle]

        def await_reads(self, posted):
            events.append("await")
            if self.fail_wait:
                raise RuntimeError("wait failed")
            self.awaited.extend(posted)

    monkeypatch.setattr(module, "NixlTransferManager", Manager)
    monkeypatch.setattr(module, "NixlReshardTransport", Transport)
    allocations = []

    def allocate(self, size: int) -> torch.Tensor:
        allocations.append(size)
        return torch.empty(size, dtype=torch.uint8)

    monkeypatch.setattr(module._NixlStagedTransfer, "_allocate_arena", allocate)

    def create_transfer() -> module._NixlStagedTransfer:
        return module._NixlStagedTransfer(
            agent_name="target",
            device_id=0,
            device=torch.device("cuda:0"),
            listen_port=None,
        )

    transfer = create_transfer()

    def manifests() -> list[bytes]:
        return [
            wrap_rendezvous_blob(
                remote["name"].encode(),
                remote["name"],
                "source:19000",
                [
                    PublishedTensor(
                        name=name,
                        dtype="torch.float32",
                        elsize=4,
                        full_shape=tuple(tensor.shape),
                        shards=[
                            PublishedShard(
                                agent_name=remote["name"],
                                device_id=0,
                                addr=tensor.data_ptr(),
                                shard_offset=(0, 0),
                                shape=tuple(tensor.shape),
                                digest=(
                                    tensor_digest(tensor)
                                    if envs.MX_RESHARD_PUBLISH_DIGEST
                                    else None
                                ),
                            )
                        ],
                    )
                    for name, tensor in sources.items()
                ],
            )
        ]

    def capture_layout(manifest):
        captures.append(manifest)
        return capture, layout

    def prepare(**kwargs) -> module._PreparedBoundedTransfer:
        return transfer.prepare_streaming(
            manifests=manifests(),
            capture_layout=capture_layout,
            trainer_snapshot=kwargs.pop(
                "trainer_snapshot",
                transfer.cached_trainer_source()
                or TrainerSourceSnapshot("mesh", 1, ()),
            ),
            **{"max_staging_bytes": 1024, "staging_device": "cpu", **kwargs},
        )

    def collect(prepared):
        metrics, installed = {}, {}
        for tensors in transfer.iter_bounded(prepared, metrics):
            installed.update({name: value.clone() for name, value in tensors.items()})
        return metrics, installed

    def new_transfer() -> None:
        nonlocal transfer
        transfer.close()
        transfer = create_transfer()
        state.transfer = transfer

    state = SimpleNamespace(
        transfer=transfer,
        new_transfer=new_transfer,
        sources=sources,
        capture=capture,
        layout=layout,
        prepare=prepare,
        collect=collect,
        captures=captures,
        transports=transports,
        events=events,
        allocations=allocations,
        remote=remote,
        manifests=manifests,
        capture_layout=capture_layout,
    )
    yield state
    transfer.close()


def _check_values(harness, installed):
    for source, name, dtype, transpose in (
        ("exact", "a.weight", torch.float32, False),
        ("full", "b.weight", torch.float32, True),
        ("convert", "c.weight", torch.bfloat16, False),
        ("exact", "d.weight", torch.float32, False),
    ):
        expected = torch.zeros(20, dtype=dtype)
        value = harness.sources[source]
        expected[2:18].copy_((value.T if transpose else value).reshape(-1))
        assert torch.equal(installed[name], expected)


@pytest.mark.parametrize("buffers", [1, 2])
@pytest.mark.parametrize("pack", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_warm_descriptors_still_transfer_new_values(
    harness, monkeypatch, buffers, pack, device
) -> None:
    monkeypatch.setenv("MX_REFIT_PACK_MODULES", str(int(pack)))
    harness.new_transfer()
    prepare = lambda: harness.prepare(staging_buffers=buffers, staging_device=device)
    first = prepare()
    cold, installed = harness.collect(first)
    _check_values(harness, installed)
    assert cold["descriptor_builds"] == first.metrics["batches"]
    assert cold["descriptor_cache_hits"] == 0
    for values in harness.sources.values():
        values.add_(3)
    for arena in harness.transfer._staging_arenas:
        arena.fill_(255)
    second = prepare()
    warm, installed = harness.collect(second)
    _check_values(harness, installed)
    assert len(harness.captures) == 1
    assert warm["descriptor_cache_hits"] == second.metrics["batches"]
    assert warm["descriptor_cache_misses"] == warm["descriptor_builds"] == 0
    assert len(harness.transports) == 1
    assert len(second.transport.posts) == 2 * second.metrics["batches"]
    assert second.transport.mem_type == ("DRAM" if device == "cpu" else "VRAM")
    assert second.transport.awaited == second.transport.posted


@pytest.mark.parametrize("change", ["address", "capture"])
def test_changed_plan_does_not_reuse_descriptors(harness, change) -> None:
    first = harness.prepare()
    harness.collect(first)
    kwargs = {"trainer_snapshot": TrainerSourceSnapshot("mesh", 2, ())}
    if change == "address":
        harness.sources["exact"] = harness.sources["exact"].clone() + 5
    elif change == "capture":
        harness.capture.copies[0] = replace(harness.capture.copies[0], dest_offset=1)
    second = harness.prepare(**kwargs)
    metrics, _ = harness.collect(second)
    assert metrics["descriptor_cache_hits"] == 0
    assert metrics["descriptor_builds"] == second.metrics["batches"]


@pytest.mark.parametrize("descriptors", [False, True])
@pytest.mark.parametrize("plan,layout", [(True, True), (False, True), (True, False)])
@pytest.mark.parametrize(
    "change", ["replace", "same_address", "reorder", "resize", "registration"]
)
def test_workspace_diagnostic_rejects_drift_before_reads(
    harness, monkeypatch, descriptors, plan, layout, change
) -> None:
    monkeypatch.setenv("MX_REFIT_DEBUG_VALIDATE_WORKSPACE", "1")
    monkeypatch.setenv("MX_REFIT_CACHE_PLAN", str(int(plan)))
    monkeypatch.setenv("MX_REFIT_CACHE_GENERATOR_LAYOUT", str(int(layout)))
    monkeypatch.setenv("MX_REFIT_CACHE_DESCRIPTORS", str(int(descriptors)))
    harness.new_transfer()
    harness.collect(harness.prepare(staging_buffers=2))
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
    posts = harness.events.count("post")
    with pytest.raises(ValueError, match="registered workspace changed"):
        prepared = harness.prepare(staging_buffers=2)
        harness.collect(prepared)
    assert harness.events.count("post") == posts
    _, installed = harness.collect(harness.prepare(staging_buffers=2))
    _check_values(harness, installed)


@pytest.mark.parametrize("buffers", [1, 2])
def test_workspace_diagnostic_between_yields_drains_without_posting_again(
    harness, monkeypatch, buffers
) -> None:
    monkeypatch.setenv("MX_REFIT_DEBUG_VALIDATE_WORKSPACE", "1")
    harness.new_transfer()
    prepared = harness.prepare(staging_buffers=buffers)
    iterator = harness.transfer.iter_bounded(prepared, {})
    next(iterator)
    posts = len(prepared.transport.posted)
    harness.transfer._staging_arenas[0] = harness.transfer._staging_arenas[0].clone()
    with pytest.raises(ValueError, match="registered workspace changed"):
        next(iterator)
    assert len(prepared.transport.posted) == posts
    assert prepared.transport.awaited == prepared.transport.posted


@pytest.mark.parametrize(
    "failure", ["coverage", "transport", "prepared", "registration"]
)
def test_failed_prepare_discards_descriptors(harness, monkeypatch, failure) -> None:
    if failure == "coverage":
        monkeypatch.setenv("MX_REFIT_CACHE_GENERATOR_LAYOUT", "0")
        harness.new_transfer()
    harness.collect(harness.prepare())
    if failure == "coverage":
        harness.layout["missing.weight"] = ((4,), torch.float32)
        expected = IncompleteRefit
    else:

        def fail(*args, **kwargs):
            raise RuntimeError("preparation failed")

        if failure == "transport":
            monkeypatch.setattr(module, "NixlReshardTransport", fail)
        elif failure == "prepared":
            monkeypatch.setattr(module, "_PreparedBoundedTransfer", fail)
        else:
            harness.transfer.reset_workspace()
            monkeypatch.setattr(harness.transfer._manager, "register_dram_buffer", fail)
        expected = RuntimeError
    with pytest.raises(expected):
        harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))


@pytest.mark.parametrize("failure", ["post", "wait", "abandon", "drain"])
def test_incomplete_iteration_discards_cache_and_preserves_drain(
    harness, failure
) -> None:
    harness.collect(harness.prepare(staging_buffers=2))
    prepared = harness.prepare(staging_buffers=2)
    previous_posts = len(prepared.transport.posted)
    if failure in ("post", "wait"):
        setattr(prepared.transport, f"fail_{failure}", True)
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            harness.collect(prepared)
    else:
        iterator = harness.transfer.iter_bounded(prepared, {})
        next(iterator)
        assert len(prepared.transport.posted) - previous_posts == 2
        if failure == "drain":
            prepared.transport.fail_wait = True
            with pytest.raises(RuntimeError, match="could not be drained"):
                iterator.close()
        else:
            iterator.close()
            assert prepared.transport.awaited == prepared.transport.posted


@pytest.mark.parametrize("cleanup", ["reset", "close"])
def test_metadata_does_not_own_arenas_or_handles(harness, cleanup) -> None:
    prepared = harness.prepare()
    harness.collect(prepared)
    entry = prepared.cached_plan
    arena_refs = [weakref.ref(arena) for arena in harness.transfer._staging_arenas]
    transport_ref = weakref.ref(prepared.transport)
    # Drop test-only observation lists and all prepared state.
    harness.transports.clear()
    del prepared
    harness.transfer._active = None
    getattr(harness.transfer, cleanup if cleanup == "close" else "reset_workspace")()
    gc.collect()
    assert all(reference() is None for reference in arena_refs)
    assert transport_ref() is None


def test_descriptor_order_duplicates_and_empty_reads_survive_reuse(
    harness, monkeypatch
) -> None:
    monkeypatch.setenv("MX_REFIT_PACK_MODULES", "1")
    harness.new_transfer()
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
    first = harness.prepare(max_staging_bytes=2048)
    assert first.metrics["batches"] == 1
    harness.collect(first)
    second = harness.prepare(max_staging_bytes=2048)
    metrics, installed = harness.collect(second)
    _check_values(harness, installed)
    assert len(builds) == 1
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


@pytest.mark.parametrize("enabled", [False, True])
def test_required_agents_are_resolved_once_per_compiled_plan(
    harness, monkeypatch, enabled
) -> None:
    monkeypatch.setenv("MX_REFIT_CACHE_PLAN", str(int(enabled)))
    harness.new_transfer()
    calls = []
    original = module._required_agent_metadata

    def record(plan, resolved) -> dict[str, bytes]:
        calls.append(plan)
        return original(plan, resolved)

    monkeypatch.setattr(module, "_required_agent_metadata", record)
    first = harness.prepare()
    cold_calls = len(calls)
    assert cold_calls == 1 + first.metrics["batches"]
    harness.collect(first)
    calls.clear()
    second = harness.prepare()
    assert len(calls) == (0 if enabled else cold_calls)
    assert second.metrics["plan_cache_hits"] == int(enabled)
    _, installed = harness.collect(second)
    _check_values(harness, installed)


@pytest.mark.parametrize("failure", ["allocation", "registration", "descriptors"])
def test_partial_receive_setup_can_be_reset_and_prepared_again(
    harness, monkeypatch, failure
) -> None:
    transfer = harness.transfer
    target, name = {
        "allocation": (transfer, "_allocate_arena"),
        "registration": (transfer._manager, "register_dram_buffer"),
        "descriptors": (transfer, "_bind_descriptors"),
    }[failure]
    original = getattr(target, name)
    attempts = 0

    def fail(*args, **kwargs) -> Any:
        nonlocal attempts
        attempts += 1
        if failure == "descriptors" or attempts == 2:
            raise RuntimeError("injected setup failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(target, name, fail)
    with pytest.raises(RuntimeError, match="injected setup failure"):
        harness.prepare(staging_buffers=2)
    transfer.reset_workspace()
    assert transfer._manager.registered == {}
    monkeypatch.setattr(target, name, original)
    _, installed = harness.collect(harness.prepare(staging_buffers=2))
    _check_values(harness, installed)


@pytest.mark.parametrize("tensor_indices", [False, True])
def test_prepared_list_indexing_replays_selected_rows_after_caller_mutation(
    harness, monkeypatch, tensor_indices
) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")
    harness.new_transfer()
    indices = torch.tensor([0, 2]) if tensor_indices else [0, 2]
    copy = harness.capture.copies[1]
    copy.op_chain = (("__getitem__", (indices,), ()),)
    copy.dest_shape = (2, 4)
    copy.dest_stride = (4, 1)
    prepared = harness.prepare()
    indices[:] = torch.tensor([1, 3]) if tensor_indices else [1, 3]
    _, installed = harness.collect(prepared)
    expected = torch.zeros(20, dtype=torch.float32)
    expected[2:10].copy_(harness.sources["full"][[0, 2]].reshape(-1))
    assert torch.equal(installed["b.weight"], expected)

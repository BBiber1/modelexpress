# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-version staged NIXL transfer for RL generator workers.

This module owns the state that makes a pull transfer correct: the selected
source manifests, the physical plan, registered destination buffers, peer
metadata, transfer completion, and verification. It deliberately does not know
how an inference engine captures its load layout or installs received weights.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import weakref
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, is_dataclass, replace
from types import MappingProxyType
from typing import Any, NamedTuple, cast

import torch
from modelexpress import envs, p2p_pb2
from modelexpress.metadata.worker_server import (
    TensorReadLease,
    prepare_tensor_read,
)
from modelexpress.nixl_transfer import (
    NIXL_DRAM_MEM_TYPE,
    NIXL_VRAM_MEM_TYPE,
    NixlTransferManager,
)
from modelexpress.refit.reshard import throughput
from modelexpress.refit.reshard.cuda_pool import classic_cuda_alloc
from modelexpress.refit.reshard.rendezvous import (
    build_sources,
    merge_shard_tables,
    unwrap_rendezvous_blob,
)
from modelexpress.refit.reshard.slice_plan import PullSegment, plan_pull
from modelexpress.refit.reshard.transfer_plan import (
    FullPullSource,
    TransferPlan,
    exact_descriptors,
    plan_transfer,
)
from modelexpress.refit.reshard.transport import ReadDescriptor
from modelexpress.refit.reshard.transport.nixl import NixlReshardTransport
from modelexpress.refit.reshard.types import (
    CaptureResult,
    IncompleteRefit,
    RecordedCopy,
    UnsupportedReshard,
    summarize_unsupported,
)
from modelexpress.refit.reshard.verify import shard_region, tensor_digest
from modelexpress.types import ManifestMismatchError, TensorDescriptor

from modelexpress_rl.inference.adapter import TrainerSourceShard
from modelexpress_rl.inference.plan import TrainerSourceSnapshot

from modelexpress_rl.inference._source_snapshot import (
    _freeze_sources,
    _ShardSnapshot,
)

from ..train.manifest import ShardChecksumKey
from .plan import StreamingSettings

# Named under modelexpress.* (not modelexpress_rl) so the per-update summary surfaces
# in the vLLM engine process, which only configures the modelexpress logger.
logger = logging.getLogger("modelexpress.reshard.staged_transfer")


def _layout_values_match(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if is_dataclass(left):
        return all(
            _layout_values_match(getattr(left, item.name), getattr(right, item.name))
            for item in fields(left)
        )
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _layout_values_match(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _layout_values_match(value, right[key]) for key, value in left.items()
        )
    return left == right


class _ConnectedAgent(NamedTuple):
    metadata: bytes
    mesh_identity: tuple[str, int]


@dataclass(frozen=True)
class _ResolvedSources:
    sources: dict
    session_to_agent: dict
    session_to_device: dict
    agent_metadata: dict[str, bytes]
    session_to_memory: dict[str, str] = field(default_factory=dict)


_SourceGeometry = tuple[str, tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class _TrainerReplica:
    source: TrainerSourceShard
    resolved: _ResolvedSources
    bindings: Mapping[_SourceGeometry, _ShardSnapshot]
    geometry: tuple



@dataclass(frozen=True)
class _WeightUpdatePlan:
    trainer_source_snapshot: TrainerSourceSnapshot
    generator_capture_snapshot: CaptureResult
    parameter_layout: Mapping
    transfer_plan: TransferPlan | _StreamingSchedule
    replicas: Mapping[tuple[str, str], _TrainerReplica] = field(default_factory=dict)
    source_slots: Mapping[_SourceGeometry, str] = field(default_factory=dict)
    replica_sources: tuple[TrainerSourceShard, ...] = field(init=False)
    required_agent_metadata: Mapping[str, bytes] | None = None
    bounded_workspace: _BoundedWorkspace | None = None

    def __post_init__(self) -> None:
        schedule = self.transfer_plan
        resolved = self.trainer_source_snapshot.resolved_metadata
        if self.required_agent_metadata is None:
            required = _required_agent_metadata(
                schedule.transfer_plan if isinstance(schedule, _StreamingSchedule) else schedule,
                resolved,
            )
            if isinstance(schedule, _StreamingSchedule):
                for batch in schedule.batches:
                    required.update(_required_agent_metadata(batch.transfer_plan, resolved))
            object.__setattr__(self, "required_agent_metadata", MappingProxyType(required))
        object.__setattr__(
            self, "replica_sources", tuple(record.source for record in self.replicas.values())
        )


@dataclass(frozen=True)
class _PreparedNixlTransfer:
    """One immutable physical plan over reusable registered destinations."""

    transfer_plan: TransferPlan
    capture: CaptureResult
    sources: Mapping
    descriptors: tuple[ReadDescriptor | _BoundedReadDescriptor, ...]
    transport: NixlReshardTransport
    metrics: dict[str, float] = field(default_factory=dict)
    checksums: Mapping[ShardChecksumKey, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _StagedNixlWeights:
    """Verified tensors ready for engine installation."""

    tensors: dict[str, torch.Tensor]
    metrics: dict[str, Any]


_StagingLayout = dict[str, tuple[tuple[int, ...], torch.dtype]]
_CaptureLayout = Callable[
    [list[tuple[str, torch.dtype, tuple[int, ...]]]],
    tuple[CaptureResult, _StagingLayout],
]


class _StagingLayouts(NamedTuple):
    """The three typed views one batch carves out of its arena, in arena order."""

    recv: _StagingLayout
    convert: _StagingLayout
    full: _StagingLayout


@dataclass(frozen=True)
class _StreamingBatch:
    capture: CaptureResult
    transfer_plan: TransferPlan
    layouts: _StagingLayouts
    nbytes: int



@dataclass(frozen=True)
class _PreparedBoundedTransfer:
    batches: tuple[_StreamingBatch, ...]
    sources: dict
    transport: NixlReshardTransport
    metrics: dict[str, float] = field(default_factory=dict)
    checksums: Mapping[ShardChecksumKey, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _StreamingSchedule:
    transfer_plan: TransferPlan
    module_batches: tuple[_StreamingBatch, ...]
    batches: tuple[_StreamingBatch, ...]
    arena_bytes: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arena_bytes", max(batch.nbytes for batch in self.batches))


class _BoundedReadDescriptor(NamedTuple):
    """Immutable addresses only; the workspace and source lease own storage."""

    session: str
    src_addr: int
    dst_addr: int
    nbytes: int


def _arena_geometry(arena: torch.Tensor) -> tuple:
    return (
        arena.data_ptr(),
        arena.untyped_storage().nbytes(),
        arena.dtype,
        arena.device,
        tuple(arena.shape),
        tuple(arena.stride()),
        arena.storage_offset(),
    )


@dataclass(frozen=True)
class _BoundedBatchBinding:
    buffers: tuple[dict[str, torch.Tensor], ...]
    descriptors: tuple[_BoundedReadDescriptor, ...]
    sources: Mapping
    full_copies: tuple[tuple[torch.Tensor, torch.Tensor, tuple], ...]
    conversion_copies: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    zero_tensors: tuple[torch.Tensor, ...]
    wire_bytes: int


@dataclass(frozen=True)
class _BoundedWorkspace:
    schedule: _StreamingSchedule
    generation: int
    arenas: tuple[tuple[weakref.ReferenceType, tuple], ...]
    batches: tuple[_BoundedBatchBinding, ...]
    staging_peak_bytes: int

    def matches(self, arenas: list[torch.Tensor]) -> bool:
        return (
            len(self.arenas) == len(arenas)
            and all(
                reference() is arena and geometry == _arena_geometry(arena)
                for (reference, geometry), arena in zip(self.arenas, arenas)
            )
        )



def _bounded_batches(
    capture: CaptureResult,
    parameter_layout: _StagingLayout,
    sources: dict,
    max_staging_bytes: int,
    *,
    complete_plan=None,
    metrics=None,
    total_staging_bytes=None,
    staging_buffers=1,
) -> tuple[_StreamingBatch, ...]:
    """Validate all owning-module batches before allocating or installing."""
    if metrics is None:
        metrics = {}
    started = time.perf_counter()
    complete = complete_plan
    if complete is None:
        complete = _plan_staged_transfer(capture, sources)
    metrics["bounded_whole_plan_s"] = time.perf_counter() - started
    metrics["bounded_whole_plan_builds"] = int(complete_plan is None)
    started = time.perf_counter()
    _NixlStagedTransfer._validate_complete(capture, parameter_layout, complete)
    metrics["bounded_whole_validation_s"] = time.perf_counter() - started
    if {copy.param_name for copy in capture.copies} - parameter_layout.keys():
        raise IncompleteRefit("bounded capture references unknown engine parameters")
    groups = {}
    for name in parameter_layout:
        groups.setdefault(name.rpartition(".")[0], {})[name] = parameter_layout[name]
    batches = []
    metrics["owner_plan_s"] = 0.0
    metrics["owner_validation_s"] = 0.0
    metrics["owner_plan_builds"] = 0
    for module, recv in groups.items():
        subset = CaptureResult(
            copies=[c for c in capture.copies if c.param_name in recv]
        )
        started = time.perf_counter()
        plan = _plan_staged_transfer(subset, sources)
        metrics["owner_plan_s"] += time.perf_counter() - started
        metrics["owner_plan_builds"] += 1
        started = time.perf_counter()
        _NixlStagedTransfer._validate_complete(subset, recv, plan)
        metrics["owner_validation_s"] += time.perf_counter() - started
        convert = {
            c.param_name: (tuple(c.dest_shape), c.src_dtype) for c in plan.converts
        }
        full = {f.src_name: (tuple(f.global_shape), f.dtype) for f in plan.full_pulls}
        layouts = _StagingLayouts(recv, convert, full)
        # Each typed view begins at a 256-byte boundary in one registered arena.
        nbytes = sum(
            ((math.prod(shape) * dtype.itemsize + 255) // 256) * 256
            for layout in layouts
            for shape, dtype in layout.values()
        )
        if nbytes > max_staging_bytes:
            if total_staging_bytes is not None and staging_buffers > 1:
                budget = (
                    f"max_staging_bytes={total_staging_bytes} split across "
                    f"staging_buffers={staging_buffers} gives {max_staging_bytes} "
                    "bytes per arena"
                )
                remedy = "lower staging_buffers or raise max_staging_bytes"
            else:
                budget = f"max_staging_bytes={max_staging_bytes}"
                remedy = "raise max_staging_bytes"
            raise IncompleteRefit(
                f"module {module!r} requires {nbytes} staging bytes, exceeds "
                f"{budget}; {remedy} (there is no CPU fallback)"
            )
        batches.append(_StreamingBatch(subset, plan, layouts, nbytes))
    if not batches:
        raise IncompleteRefit("bounded refit has no engine parameters")
    return tuple(batches)


def _pack_bounded_batches(
    batches: tuple[_StreamingBatch, ...], max_staging_bytes: int
) -> tuple[_StreamingBatch, ...]:
    """Pack complete modules without changing source READ ranges.

    Owning-module batches are the unit of correctness; this only coalesces
    neighbours that fit the same arena together, so each packed batch still
    installs whole modules and reads exactly the bytes the unpacked plan read.
    """

    def merge(group: list[_StreamingBatch]) -> _StreamingBatch:
        capture = CaptureResult(
            copies=[copy for batch in group for copy in batch.capture.copies]
        )
        plan = TransferPlan()
        layouts = _StagingLayouts({}, {}, {})
        for batch in group:
            _merge_plan(plan, batch.transfer_plan)
            for layout, incoming in zip(layouts, batch.layouts, strict=True):
                layout.update(incoming)
        return _StreamingBatch(
            capture, plan, layouts, sum(batch.nbytes for batch in group)
        )

    packed = []
    current = []
    current_bytes = 0
    full_sources = set()
    for batch in batches:
        incoming_full = set(batch.layouts.full)
        # Two modules pulling the same complete source would need one staging
        # slot for two distinct writes, so they must stay in separate batches.
        if current and (
            current_bytes + batch.nbytes > max_staging_bytes
            or full_sources & incoming_full
        ):
            packed.append(merge(current))
            current = []
            current_bytes = 0
            full_sources = set()
        current.append(batch)
        current_bytes += batch.nbytes
        full_sources.update(incoming_full)
    if current:
        packed.append(merge(current))
    return tuple(packed)


def _resolve_sources(manifests: list[bytes], *, metrics=None) -> _ResolvedSources:
    if not manifests:
        raise ValueError("at least one source manifest is required")
    if metrics is None:
        metrics = {}
    started = time.perf_counter()
    payloads = [unwrap_rendezvous_blob(manifest) for manifest in manifests]
    agents = [payload.agent_name for payload in payloads]
    if len(set(agents)) != len(agents):
        raise ValueError("source manifests contain duplicate NIXL agents")
    metrics["source_decode_s"] = time.perf_counter() - started
    started = time.perf_counter()
    merged = merge_shard_tables([payload.tensors for payload in payloads])
    metrics["source_merge_s"] = time.perf_counter() - started
    started = time.perf_counter()
    session_to_memory = {}
    sources, session_to_agent, session_to_device = build_sources(
        merged, session_to_memory=session_to_memory
    )
    metrics["source_build_s"] = time.perf_counter() - started
    return _ResolvedSources(
        sources=sources,
        session_to_agent=session_to_agent,
        session_to_device=session_to_device,
        session_to_memory=session_to_memory,
        agent_metadata={
            payload.agent_name: payload.agent_metadata for payload in payloads
        },
    )


def _trainer_replica(
    source: TrainerSourceShard, resolved: _ResolvedSources | None = None
) -> _TrainerReplica:
    if resolved is None:
        resolved = _resolve_sources([source.metadata])
        frozen_sources = _freeze_sources(resolved.sources)
        if frozen_sources is None:
            raise RuntimeError("trainer replica metadata cannot be isolated")
        resolved = replace(resolved, sources=frozen_sources)
    resolved = replace(
        resolved,
        session_to_agent=MappingProxyType(dict(resolved.session_to_agent)),
        session_to_device=MappingProxyType(dict(resolved.session_to_device)),
        session_to_memory=MappingProxyType(dict(resolved.session_to_memory)),
        agent_metadata=MappingProxyType(dict(resolved.agent_metadata)),
    )
    bindings = {}
    geometry = []
    for name, tensor in resolved.sources.items():
        boxes = []
        for shard in tensor.shards:
            box = (tuple(shard.shard_offset), tuple(shard.shape))
            key = (name, *box)
            if key in bindings:
                raise RuntimeError(f"trainer replica contains ambiguous geometry for {name!r}")
            bindings[key] = shard
            boxes.append(box)
        geometry.append(
            (name, tensor.dtype, tuple(tensor.global_shape), tensor.elsize, tuple(sorted(boxes)))
        )
    return _TrainerReplica(
        source, resolved, MappingProxyType(bindings), tuple(sorted(geometry))
    )


def _bind_transfer_plan(
    plan: TransferPlan,
    source_slots: Mapping[_SourceGeometry, str],
    selected: Mapping[str, _TrainerReplica],
) -> TransferPlan:
    def bind(segment: PullSegment) -> PullSegment:
        if segment._source_read is None:
            raise RuntimeError("transfer read has no source shard provenance")
        name, offset, shape, relative_offset = segment._source_read
        key = (name, offset, shape)
        replica = selected[source_slots[key]]
        shard = replica.bindings[key]
        return replace(segment, session=shard.session, src_addr=shard.addr + relative_offset)

    return replace(
        plan,
        segments=[bind(segment) for segment in plan.segments],
        converts=[
            replace(convert, segments=[bind(segment) for segment in convert.segments])
            for convert in plan.converts
        ],
        full_pulls=[
            replace(full, segments=[bind(segment) for segment in full.segments])
            for full in plan.full_pulls
        ],
    )


def _bind_streaming_schedule(
    schedule: _StreamingSchedule,
    source_slots: Mapping[_SourceGeometry, str],
    selected: Mapping[str, _TrainerReplica],
) -> _StreamingSchedule:
    def bind(batch: _StreamingBatch) -> _StreamingBatch:
        return replace(
            batch, transfer_plan=_bind_transfer_plan(batch.transfer_plan, source_slots, selected)
        )

    return _StreamingSchedule(
        _bind_transfer_plan(schedule.transfer_plan, source_slots, selected),
        schedule.module_batches,
        tuple(bind(batch) for batch in schedule.batches),
    )



def _required_agent_metadata(
    plan: TransferPlan, resolved: _ResolvedSources
) -> dict[str, bytes]:
    sessions = plan.sessions()
    missing_sessions = sorted(sessions - set(resolved.session_to_agent))
    if missing_sessions:
        raise RuntimeError(
            f"transfer plan references unknown source sessions: {missing_sessions[:10]}"
        )
    needed = {resolved.session_to_agent[session] for session in sessions}
    missing = sorted(needed - set(resolved.agent_metadata))
    if missing:
        raise RuntimeError(
            "transfer plan references source agents without NIXL metadata: "
            f"{missing[:10]}"
        )
    return {
        agent: metadata
        for agent, metadata in resolved.agent_metadata.items()
        if agent in needed
    }


def _load_agent_metadata(
    manager: NixlTransferManager, metadata_by_agent: dict[str, bytes]
) -> None:
    """Load the exact source registrations carried by the version manifests."""
    for expected_agent, metadata in metadata_by_agent.items():
        loaded_agent = manager.add_remote_agent(metadata)
        if isinstance(loaded_agent, bytes):
            loaded_agent = loaded_agent.decode("utf-8")
        if loaded_agent != expected_agent:
            raise RuntimeError(
                "NIXL metadata agent does not match its manifest: "
                f"expected {expected_agent!r}, got {loaded_agent!r}"
            )


def _replay_ops(tensor: torch.Tensor, op_chain: tuple) -> torch.Tensor:
    value = tensor
    for op_name, args, frozen_kwargs in op_chain:
        kwargs = dict(frozen_kwargs)
        if op_name == "__getitem__":
            value = value.__getitem__(*args)
        else:
            value = getattr(value, op_name)(*args, **kwargs)
    return value


def _row_major_strides(shape: tuple) -> tuple:
    strides = []
    stride = 1
    for extent in reversed(shape):
        strides.append(stride)
        stride *= int(extent)
    return tuple(reversed(strides))


def _merge_plan(target: TransferPlan, source: TransferPlan) -> None:
    target.segments.extend(source.segments)
    target.converts.extend(source.converts)
    target.full_pulls.extend(source.full_pulls)
    target.unbounded_sources.extend(source.unbounded_sources)
    for name in source.fallback:
        if name not in target.fallback:
            target.fallback.append(name)
    target.exact_descriptor_count += source.exact_descriptor_count
    target.exact_bytes += source.exact_bytes


def _plan_staged_transfer(
    capture: CaptureResult, sources: dict
) -> TransferPlan:
    """Plan reads for each source.

    Default: minimal slice reads via plan_transfer (a partial read of a shard cannot
    be whole-shard digest-verified, so correctness rests on the coverage gate). Under
    MX_RESHARD_PUBLISH_DIGEST (verification mode): reconstruct every source that isn't
    a whole-tensor identity copy as a full pull, so _verify has a complete shard to
    digest-check.
    """
    if not envs.MX_RESHARD_PUBLISH_DIGEST:
        return plan_transfer(capture, sources)

    result = TransferPlan()
    copies_by_source: dict[str, list[RecordedCopy]] = {}
    for copy in capture.copies:
        copies_by_source.setdefault(copy.src_name, []).append(copy)

    for name in capture.unsupported:
        if name not in result.fallback:
            result.fallback.append(name)

    for name, source in sources.items():
        copies = copies_by_source.pop(name, [])
        if not copies:
            continue
        directly_recoverable = any(
            not copy.op_chain and tuple(copy.dest_shape) == tuple(source.global_shape)
            for copy in copies
        )
        if directly_recoverable:
            _merge_plan(
                result,
                plan_transfer(CaptureResult(copies=copies), {name: source}),
            )
            continue

        identity = RecordedCopy(
            src_name=name,
            op_chain=(),
            param_name=name,
            dest_offset=0,
            dest_shape=tuple(source.global_shape),
            dest_stride=_row_major_strides(source.global_shape),
            dest_dtype=source.dtype,
        )
        try:
            segments = plan_pull(
                identity,
                source.global_shape,
                source.dtype,
                source.elsize,
                source.shards,
            )
        except UnsupportedReshard as error:
            raise UnsupportedReshard(
                f"{name}: strict staged verification cannot reconstruct the "
                "complete published source"
            ) from error
        result.full_pulls.append(
            FullPullSource(
                src_name=name,
                global_shape=tuple(source.global_shape),
                dtype=source.dtype,
                elsize=source.elsize,
                segments=segments,
                copies=copies,
            )
        )
        result.exact_descriptor_count += len(segments)
        result.exact_bytes += sum(segment.nbytes for segment in segments)

    for name in copies_by_source:
        if name not in result.fallback:
            result.fallback.append(name)
    return result


class _NixlStagedTransfer:
    """Own the complete prepare-and-stage lifecycle for one generator rank."""

    def __init__(
        self,
        *,
        device_id: int,
        device: torch.device,
        agent_name: str | None = None,
        listen_port: int | None = None,
        timeout_seconds: float | None = None,
        manager: NixlTransferManager | None = None,
        streaming: StreamingSettings | None = None,
    ) -> None:
        if streaming is not None and device.type != "cuda":
            raise ValueError("bounded NIXL staging requires a CUDA device")
        self._streaming = streaming
        self._buffer_budget = (
            None
            if streaming is None
            else streaming.max_staging_bytes // streaming.staging_buffers
        )
        self._device_id = device_id
        self._device = device
        self._timeout = float(
            envs.MX_TRANSFER_TIMEOUT if timeout_seconds is None else timeout_seconds
        )
        self._owns_manager = manager is None
        if manager is None:
            if agent_name is None:
                raise ValueError("an owned NIXL manager requires an agent name")
            manager = NixlTransferManager(
                agent_name=agent_name,
                device_id=device_id,
                listen_port=listen_port,
            )
        self._manager = manager
        if self._owns_manager:
            try:
                self._manager.initialize()
            except Exception:
                self._manager.shutdown()
                raise
        # Canonical engine-layout staging buffers. Exact slices land directly
        # here; reconstructed or converted values are copied here before these
        # buffers are verified, installed, and advertised to peer generators.
        self._recv_buffers: dict[str, torch.Tensor] = {}
        # Wire-dtype staging for sources whose dtype differs from the engine
        # parameter. RDMA writes here first, then stage() casts into recv buffers.
        self._convert_buffers: dict[str, torch.Tensor] = {}
        # Complete contiguous source tensors used when captured transforms must
        # be replayed locally, or when direct slicing exceeds the descriptor
        # budget. stage() reconstructs each source here, then copies its derived
        # views into the canonical receive buffers.
        self._full_buffers: dict[str, torch.Tensor] = {}
        self._registered_recv_params: set[str] = set()
        self._convert_registered = False
        self._full_registered = False
        self._active: _PreparedNixlTransfer | _PreparedBoundedTransfer | None = None
        self._loaded_agent_metadata: dict[str, _ConnectedAgent] = {}
        self._closed = False
        # Bounded staging: one or two byte arenas on CUDA or pinned host memory.
        # Host arenas are registered as NIXL DRAM and tracked here so they can be
        # deregistered before the agent shuts down.
        self._staging_arenas: list[torch.Tensor] = []
        self._staging_registrations: list[Any] = []
        self._staging_device: torch.device | None = None
        self._manager_ready = True
        self._weight_update_plan: _WeightUpdatePlan | None = None
        self._full_copy_descriptors: tuple[ReadDescriptor, ...] | None = None
        self._debug_validate_plan = envs.MX_REFIT_DEBUG_VALIDATE_PLAN
        self._debug_validate_layout = envs.MX_REFIT_DEBUG_VALIDATE_GENERATOR_LAYOUT
        self._pack_modules = envs.MX_REFIT_PACK_MODULES
        self._plan_compile_lock = threading.Lock()
        self._workspace_generation = 0
        self._debug_validate_workspace = envs.MX_REFIT_DEBUG_VALIDATE_WORKSPACE

    @property
    def _bounded_arena(self) -> torch.Tensor | None:
        return self._staging_arenas[0] if self._staging_arenas else None

    @_bounded_arena.setter
    def _bounded_arena(self, value: torch.Tensor | None) -> None:
        self._invalidate_workspace_binding()
        self._staging_arenas = [] if value is None else [value]

    def _invalidate_workspace_binding(self) -> None:
        self._workspace_generation += 1

    def _release_staging_registrations(self) -> None:
        self._invalidate_workspace_binding()
        registrations, self._staging_registrations = self._staging_registrations, []
        for registration in registrations:
            self._manager.deregister_memory(registration)

    def _allocate_arena(self, nbytes: int) -> torch.Tensor:
        assert self._staging_device is not None
        if self._staging_device.type == "cpu":
            # Pinned so the NIC can register it and the H2D commit is a DMA.
            return torch.empty(
                nbytes, dtype=torch.uint8, pin_memory=torch.cuda.is_available()
            )
        with classic_cuda_alloc():
            return torch.empty(nbytes, dtype=torch.uint8, device=self._device)

    def reset_workspace(self) -> None:
        """Discard released or failed preparation after disconnecting its agent."""
        self._invalidate_workspace_binding()
        if not self._owns_manager:
            # A shared agent still serves its owner; tearing it down here would
            # deregister memory we do not own. Only a transfer-owned agent can
            # be cycled to release staging storage safely.
            raise RuntimeError(
                "resetting staging storage requires a transfer-owned NIXL agent; "
                "restart the generator engine"
            )
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        self._release_staging_registrations()
        self._manager.shutdown()
        self._active = None
        self._recv_buffers.clear()
        self._convert_buffers.clear()
        self._full_buffers.clear()
        self._staging_arenas.clear()
        self._staging_device = None
        self._registered_recv_params.clear()
        self._convert_registered = False
        self._full_registered = False
        self._loaded_agent_metadata.clear()
        self._manager_ready = False
        self._weight_update_plan = None
        self._full_copy_descriptors = None

    def _ensure_manager_initialized(self) -> None:
        if not self._manager_ready and self._owns_manager:
            self._native_setup_started = True
            try:
                self._manager.initialize()
            except Exception:
                self._manager.shutdown()
                raise
            self._manager_ready = True

    def _resolve_metadata(
        self, manifests: list[bytes], metrics: dict
    ) -> _ResolvedSources:
        if any(not isinstance(blob, bytes) for blob in manifests):
            raise TypeError("cached source manifests must be immutable bytes")
        metrics.update(
            source_cache_enabled=1,
            source_cache_hits=0,
            source_cache_misses=0,
            source_cache_lookup_s=0.0,
            source_decode_s=0.0,
            source_merge_s=0.0,
            source_build_s=0.0,
            source_manifest_bytes=sum(len(blob) for blob in manifests),
        )
        started = time.perf_counter()
        previous = self._weight_update_plan
        hit = previous is not None and tuple(source.metadata for source in previous.trainer_source_snapshot.shards) == tuple(manifests)
        metrics["source_cache_lookup_s"] = time.perf_counter() - started
        metrics["source_cache_hits"] = int(hit)
        metrics["source_cache_misses"] = int(not hit)
        if hit:
            return previous.trainer_source_snapshot.resolved_metadata
        resolved = _resolve_sources(manifests, metrics=metrics)
        started = time.perf_counter()
        frozen_sources = _freeze_sources(resolved.sources)
        if frozen_sources is not None:
            resolved = replace(resolved, sources=frozen_sources)
        metrics["source_build_s"] += time.perf_counter() - started
        resolved = replace(
            resolved,
            session_to_agent=MappingProxyType(dict(resolved.session_to_agent)),
            session_to_device=MappingProxyType(dict(resolved.session_to_device)),
            session_to_memory=MappingProxyType(dict(resolved.session_to_memory)),
            agent_metadata=MappingProxyType(dict(resolved.agent_metadata)),
        )
        return resolved

    def _resolve_layout(
        self,
        manifests: list[bytes],
        capture_layout: _CaptureLayout,
        metrics: dict[str, float],
        trainer_snapshot: TrainerSourceSnapshot,
    ) -> tuple[TrainerSourceSnapshot, CaptureResult, _StagingLayout]:
        started = time.perf_counter()
        resolved = self._resolve_metadata(manifests, metrics)
        trainer = replace(trainer_snapshot, resolved_metadata=resolved)
        previous = self._weight_update_plan
        source_schema = tuple(
            (name, source.dtype, tuple(source.global_shape))
            for name, source in resolved.sources.items()
        )
        metrics["source_metadata_s"] = time.perf_counter() - started
        started = time.perf_counter()
        capture, parameter_layout = capture_layout(list(source_schema))
        if (
            self._debug_validate_layout
            and previous is not None
            and trainer.physical_fingerprint == previous.trainer_source_snapshot.physical_fingerprint
            and (
                not _layout_values_match(previous.generator_capture_snapshot, capture)
                or previous.parameter_layout != parameter_layout
            )
        ):
            raise RuntimeError("engine load layout changed for an unchanged source")
        metrics["layout_capture_s"] = time.perf_counter() - started
        return trainer, capture, parameter_layout

    def cached_trainer_source(self) -> TrainerSourceSnapshot | None:
        plan = self._weight_update_plan
        if (
            plan is None
            or self._closed
            or self._debug_validate_plan
            or self._debug_validate_layout
        ):
            return None
        return plan.trainer_source_snapshot

    def cached_trainer_replicas(
        self, mesh_id: str, generation: int
    ) -> tuple[TrainerSourceShard, ...]:
        plan = self._weight_update_plan
        if (
            plan is None
            or self._closed
            or plan.trainer_source_snapshot.mesh_id != mesh_id
            or plan.trainer_source_snapshot.mesh_generation != generation
        ):
            return ()
        return plan.replica_sources


    def _retain_plan(
        self,
        trainer: TrainerSourceSnapshot,
        capture: CaptureResult,
        parameter_layout: Mapping,
        transfer_plan: TransferPlan | _StreamingSchedule,
    ) -> _WeightUpdatePlan:
        previous = self._weight_update_plan
        replicas = dict(previous.replicas) if (
            previous is not None
            and trainer.mesh_id == previous.trainer_source_snapshot.mesh_id
            and trainer.mesh_generation == previous.trainer_source_snapshot.mesh_generation
        ) else {}
        selected = []
        for source in trainer.shards:
            identity = (source.source_slot_id, source.worker_id)
            record = replicas.get(identity)
            if record is not None:
                if record.source.physical_fingerprint != source.physical_fingerprint:
                    raise RuntimeError("known trainer replica metadata changed within a mesh generation")
            else:
                record = _trainer_replica(
                    source, trainer.resolved_metadata if len(trainer.shards) == 1 else None
                )
                replicas[identity] = record
            selected.append(record)
        source_slots = {}
        for record in selected:
            for geometry in record.bindings:
                source_slots.setdefault(geometry, record.source.source_slot_id)
        initial = (
            transfer_plan.transfer_plan
            if isinstance(transfer_plan, _StreamingSchedule) else transfer_plan
        )
        by_slot = {record.source.source_slot_id: record for record in selected}
        for segment in initial._all_segments():
            if segment._source_read is None:
                raise RuntimeError("transfer read has no source shard provenance")
            name, offset, shape, relative_offset = segment._source_read
            key = (name, offset, shape)
            shard = by_slot[source_slots[key]].bindings[key]
            if segment.session != shard.session or segment.src_addr != shard.addr + relative_offset:
                raise RuntimeError("transfer read does not match its selected source shard")
        return _WeightUpdatePlan(
            trainer, capture, parameter_layout, transfer_plan,
            MappingProxyType(replicas), MappingProxyType(source_slots),
        )


    def _try_rebind_plan(
        self,
        trainer: TrainerSourceSnapshot,
    ) -> _WeightUpdatePlan | None:
        previous = self._weight_update_plan
        if (
            previous is None
            or self._debug_validate_plan
            or self._debug_validate_layout
            or trainer.mesh_id != previous.trainer_source_snapshot.mesh_id
            or trainer.mesh_generation != previous.trainer_source_snapshot.mesh_generation
            or {source.source_slot_id for source in trainer.shards}
            != {source.source_slot_id for source in previous.trainer_source_snapshot.shards}
        ):
            return None
        replicas = dict(previous.replicas)
        selected = {}
        for source in trainer.shards:
            identity = (source.source_slot_id, source.worker_id)
            record = replicas.get(identity)
            if record is not None:
                if record.source.physical_fingerprint != source.physical_fingerprint:
                    raise RuntimeError("known trainer replica metadata changed within a mesh generation")
            else:
                record = _trainer_replica(source)
                original = next(
                    item for item in previous.replicas.values()
                    if item.source.source_slot_id == source.source_slot_id
                )
                if record.geometry != original.geometry:
                    raise RuntimeError("trainer replica geometry differs from the compiled plan")
                replicas[identity] = record
            selected[source.source_slot_id] = record
        sources = {}
        session_to_agent, session_to_device, session_to_memory, agent_metadata = {}, {}, {}, {}
        for slot, record in selected.items():
            for name, tensor in record.resolved.sources.items():
                owned = tuple(
                    shard for shard in tensor.shards
                    if previous.source_slots[(name, tuple(shard.shard_offset), tuple(shard.shape))] == slot
                )
                if not owned:
                    continue
                current = sources.get(name)
                sources[name] = (
                    tensor._replace(shards=owned) if current is None
                    else current._replace(shards=current.shards + owned)
                )
            if set(agent_metadata) & record.resolved.agent_metadata.keys():
                raise ValueError("source manifests contain duplicate NIXL agents")
            for target, incoming in (
                (session_to_agent, record.resolved.session_to_agent),
                (session_to_device, record.resolved.session_to_device),
                (session_to_memory, record.resolved.session_to_memory),
            ):
                if any(key in target and target[key] != value for key, value in incoming.items()):
                    raise RuntimeError("trainer replicas contain conflicting source sessions")
                target.update(incoming)
            agent_metadata.update(record.resolved.agent_metadata)
        sources = {name: sources[name] for name in previous.trainer_source_snapshot.resolved_metadata.sources}
        resolved = _ResolvedSources(
            MappingProxyType(sources), MappingProxyType(session_to_agent),
            MappingProxyType(session_to_device), MappingProxyType(agent_metadata),
            MappingProxyType(session_to_memory),
        )
        trainer = replace(trainer, resolved_metadata=resolved)
        plan = previous.transfer_plan
        bound = (
            _bind_streaming_schedule(plan, previous.source_slots, selected)
            if isinstance(plan, _StreamingSchedule)
            else _bind_transfer_plan(plan, previous.source_slots, selected)
        )
        return _WeightUpdatePlan(
            trainer, previous.generator_capture_snapshot, previous.parameter_layout, bound,
            MappingProxyType(replicas), previous.source_slots,
        )


    def _can_reuse_plan(self, trainer: TrainerSourceSnapshot) -> bool:
        previous = self._weight_update_plan
        return (
            previous is not None
            and trainer is previous.trainer_source_snapshot
            and not self._debug_validate_plan
            and not self._debug_validate_layout
        )

    def _compile_streaming(
        self,
        trainer: TrainerSourceSnapshot,
        capture: CaptureResult,
        parameter_layout: Mapping,
        metrics: dict,
    ) -> _StreamingSchedule:
        if not self._plan_compile_lock.acquire(blocking=False):
            raise RuntimeError("bounded plan compilation is already in progress")
        try:
            metrics.update(
                plan_cache_enabled=1,
                plan_cache_hits=0,
                plan_cache_misses=0,
                plan_cache_lookup_s=0.0,
                plan_cache_validate_s=0.0,
                initial_whole_plan_s=0.0,
                initial_whole_validation_s=0.0,
                bounded_whole_plan_s=0.0,
                bounded_whole_plan_builds=0,
                bounded_whole_validation_s=0.0,
                owner_plan_s=0.0,
                owner_validation_s=0.0,
                owner_plan_builds=0,
            )
            started = time.perf_counter()
            previous = self._weight_update_plan
            hit = (
                previous is not None
                and isinstance(previous.transfer_plan, _StreamingSchedule)
                and previous.trainer_source_snapshot.physical_fingerprint == trainer.physical_fingerprint
                and tuple(source.metadata for source in trainer.shards)
                == tuple(source.metadata for source in previous.trainer_source_snapshot.shards)
                and _layout_values_match(previous.generator_capture_snapshot, capture)
                and previous.parameter_layout == parameter_layout
            )
            metrics["plan_cache_lookup_s"] = time.perf_counter() - started
            metrics["plan_cache_hits"] = int(hit)
            metrics["plan_cache_misses"] = int(not hit)
            if hit:
                compiled = previous.transfer_plan
                if self._debug_validate_plan:
                    started = time.perf_counter()
                    _NixlStagedTransfer._validate_complete(
                        capture, parameter_layout, compiled.transfer_plan
                    )
                    metrics["bounded_whole_validation_s"] = time.perf_counter() - started
                    owner_started = time.perf_counter()
                    for batch in compiled.module_batches:
                        _NixlStagedTransfer._validate_complete(
                            batch.capture, batch.layouts[0], batch.transfer_plan
                        )
                    metrics["owner_validation_s"] = time.perf_counter() - owner_started
                    metrics["plan_cache_validate_s"] = time.perf_counter() - started
                logger.info("reusing bounded physical plan")
                return compiled
            started = time.perf_counter()
            plan = _plan_staged_transfer(capture, trainer.resolved_metadata.sources)
            metrics["initial_whole_plan_s"] = time.perf_counter() - started
            modules = _bounded_batches(
                capture,
                parameter_layout,
                trainer.resolved_metadata.sources,
                self._buffer_budget,
                complete_plan=plan,
                metrics=metrics,
                total_staging_bytes=self._streaming.max_staging_bytes,
                staging_buffers=self._streaming.staging_buffers,
            )
            batches = (
                _pack_bounded_batches(modules, self._buffer_budget)
                if self._pack_modules else modules
            )
            compiled = _StreamingSchedule(plan, modules, batches)
            logger.info("compiled bounded physical plan")
            return compiled

        finally:
            self._plan_compile_lock.release()

    def _connect_sources(
        self,
        resolved: _ResolvedSources,
        required_metadata: Mapping[str, bytes],
        *,
        host_staging: bool = False,
        mesh_identity: tuple[str, int],
        recreate_workspace: bool = False,
    ) -> NixlReshardTransport:
        changed = {
            agent: metadata for agent, metadata in required_metadata.items()
            if agent not in self._loaded_agent_metadata
            or self._loaded_agent_metadata[agent].metadata != metadata
        }
        conflicting = sorted(agent for agent in changed if agent in self._loaded_agent_metadata)
        unsafe = [
            agent for agent in conflicting
            if self._loaded_agent_metadata[agent].mesh_identity == mesh_identity
        ]
        if unsafe:
            raise RuntimeError(
                "NIXL metadata changed for an already connected source agent: "
                f"{unsafe[:10]}"
            )
        if conflicting or recreate_workspace:
            self.reset_workspace()
            changed = dict(required_metadata)
        self._ensure_manager_initialized()
        if changed:
            self._native_setup_started = True
        _load_agent_metadata(self._manager, changed)
        for agent, metadata in required_metadata.items():
            previous = self._loaded_agent_metadata.get(agent)
            if previous is None or previous.mesh_identity != mesh_identity:
                self._loaded_agent_metadata[agent] = _ConnectedAgent(metadata, mesh_identity)
        return NixlReshardTransport(
            self._manager,
            resolved.session_to_agent,
            resolved.session_to_device,
            timeout_seconds=self._timeout,
            local_mem_type=NIXL_DRAM_MEM_TYPE if host_staging else NIXL_VRAM_MEM_TYPE,
            session_to_memory=resolved.session_to_memory,
        )

    @contextmanager
    def _preparing(self) -> Iterator[None]:
        if self._closed:
            raise RuntimeError("NIXL staged transfer is closed")
        self._native_setup_started = False
        try:
            yield
        except Exception:
            self._weight_update_plan = None
            self._full_copy_descriptors = None
            if self._native_setup_started:
                try:
                    self.reset_workspace()
                except Exception as cleanup_error:
                    mode = "streaming" if self._streaming is not None else "full-copy"
                    raise ValueError(
                        f"failed to reset {mode} preparation; restart the generator engine"
                    ) from cleanup_error
            raise
        finally:
            self._native_setup_started = False

    def _publish_prepared(
        self,
        cached: _WeightUpdatePlan,
        prepared: _PreparedNixlTransfer | _PreparedBoundedTransfer,
    ) -> None:
        self._weight_update_plan = cached
        self._active = prepared

    def prepare_full_copy(
        self,
        *,
        manifests: list[bytes],
        capture_layout: _CaptureLayout,
        trainer_snapshot: TrainerSourceSnapshot,
        checksums: Mapping[ShardChecksumKey, str] | None = None,
    ) -> _PreparedNixlTransfer:
        """Create a version-scoped transfer over one reusable full-copy plan."""
        if self._streaming is not None:
            raise RuntimeError("this transfer owns bounded staging storage")
        with self._preparing():
            previous = self._weight_update_plan
            metrics = {}
            warm = self._can_reuse_plan(trainer_snapshot)
            rebound = None if warm else self._try_rebind_plan(trainer_snapshot)
            layout = (
                self._resolve_layout(manifests, capture_layout, metrics, trainer_snapshot)
                if self._debug_validate_layout and not warm and rebound is None else None
            )
            reusable = (
                previous is not None
                and isinstance(previous.transfer_plan, TransferPlan)
                and (warm or previous.trainer_source_snapshot.physical_fingerprint
                     == trainer_snapshot.physical_fingerprint)
                and (layout is None or (
                    _layout_values_match(previous.generator_capture_snapshot, layout[1])
                    and previous.parameter_layout == layout[2]
                ))
            )
            metrics.update(
                plan_cache_hits=int(reusable), plan_cache_misses=int(not reusable)
            )
            if rebound is not None:
                trainer = rebound.trainer_source_snapshot
                capture = rebound.generator_capture_snapshot
                parameter_layout = rebound.parameter_layout
                plan = rebound.transfer_plan
                metrics.update(plan_cache_hits=1, plan_cache_misses=0, replica_rebindings=1)
            elif reusable:
                trainer = previous.trainer_source_snapshot
                capture = previous.generator_capture_snapshot
                parameter_layout = previous.parameter_layout
                plan = previous.transfer_plan
                if layout is not None:
                    trainer, capture, parameter_layout = layout
                if self._debug_validate_plan:
                    self._validate_complete(capture, dict(parameter_layout), plan)
            else:
                trainer, capture, parameter_layout = (
                    layout
                    or self._resolve_layout(
                        manifests, capture_layout, metrics, trainer_snapshot
                    )
                )
                started = time.perf_counter()
                plan = _plan_staged_transfer(capture, trainer.resolved_metadata.sources)
                metrics["initial_whole_plan_s"] = time.perf_counter() - started
                validation_started = time.perf_counter()
                self._validate_complete(capture, dict(parameter_layout), plan)
                metrics["initial_whole_validation_s"] = (
                    time.perf_counter() - validation_started
                )
                metrics["transfer_planning_s"] = time.perf_counter() - started
            resolved = trainer.resolved_metadata
            cached = previous if warm else (rebound or self._retain_plan(
                trainer, capture, parameter_layout, plan
            ))
            transport = self._connect_sources(
                resolved, cached.required_agent_metadata,
                mesh_identity=(trainer.mesh_id, trainer.mesh_generation),
            )
            self._ensure_workspace(plan, parameter_layout)
            if rebound is not None or not reusable or self._full_copy_descriptors is None:
                self._full_copy_descriptors = tuple(self._descriptors(plan))
            prepared = _PreparedNixlTransfer(
                transfer_plan=plan,
                capture=capture,
                sources={
                    copy.src_name: resolved.sources[copy.src_name]
                    for copy in capture.copies
                    if copy.src_name in resolved.sources
                },
                descriptors=self._full_copy_descriptors,
                transport=transport,
                metrics=metrics,
                checksums=checksums if checksums is not None else MappingProxyType({}),
            )
            self._publish_prepared(cached, prepared)
            return prepared


    def prepare_streaming(
        self,
        *,
        manifests: list[bytes],
        capture_layout: _CaptureLayout,
        trainer_snapshot: TrainerSourceSnapshot,
        checksums: Mapping[ShardChecksumKey, str] | None = None,
    ) -> _PreparedBoundedTransfer:
        """Create fresh deferred reads over one reusable bounded pull plan."""
        streaming = self._streaming
        if streaming is None:
            raise RuntimeError("this transfer owns full-copy staging storage")
        staging_device = streaming.staging_device
        assert self._buffer_budget is not None
        with self._preparing():
            previous = self._weight_update_plan
            metrics = {}
            warm = (
                previous is not None
                and isinstance(previous.transfer_plan, _StreamingSchedule)
                and self._can_reuse_plan(trainer_snapshot)
            )
            rebound = None if warm else self._try_rebind_plan(trainer_snapshot)
            started = time.perf_counter()
            if rebound is not None:
                trainer = rebound.trainer_source_snapshot
                capture = rebound.generator_capture_snapshot
                parameter_layout = rebound.parameter_layout
                compiled = rebound.transfer_plan
                metrics.update(plan_cache_enabled=1, plan_cache_hits=1, plan_cache_misses=0, replica_rebindings=1)
            elif warm:
                trainer = previous.trainer_source_snapshot
                capture = previous.generator_capture_snapshot
                parameter_layout = previous.parameter_layout
                compiled = previous.transfer_plan
                metrics.update(plan_cache_enabled=1, plan_cache_hits=1, plan_cache_misses=0)
            else:
                trainer, capture, parameter_layout = (
                    self._resolve_layout(
                        manifests,
                        capture_layout,
                        metrics,
                        trainer_snapshot,
                    )
                )
                started = time.perf_counter()
                compiled = self._compile_streaming(
                    trainer, capture, parameter_layout, metrics
                )
            metrics["transfer_planning_s"] = time.perf_counter() - started
            started = time.perf_counter()
            resolved = trainer.resolved_metadata
            cached = previous if warm else (rebound or self._retain_plan(
                trainer, capture, parameter_layout, compiled
            ))
            transport = self._connect_sources(
                resolved, cached.required_agent_metadata, host_staging=staging_device == "cpu",
                mesh_identity=(trainer.mesh_id, trainer.mesh_generation),
                recreate_workspace=bool(
                    self._staging_arenas
                    and self._staging_arenas[0].numel() < compiled.arena_bytes
                ),
            )
            if (
                cached.bounded_workspace is None
                or cached.bounded_workspace.generation != self._workspace_generation
            ):
                self._prepare_arenas(compiled.arena_bytes)
            metrics["connection_registration_s"] = time.perf_counter() - started
            binding = cached.bounded_workspace
            reused_binding = (
                binding is not None
                and binding.schedule is compiled
                and binding.generation == self._workspace_generation
            )
            if not reused_binding:
                binding = self._bind_bounded_workspace(compiled, resolved.sources)
                cached = replace(cached, bounded_workspace=binding)
            metrics["descriptor_binding_reused"] = int(reused_binding)
            metrics["descriptor_builds"] = 0 if reused_binding else len(compiled.batches)
            prepared = _PreparedBoundedTransfer(
                compiled.batches,
                resolved.sources,
                transport,
                metrics,
                checksums if checksums is not None else MappingProxyType({}),
            )
            self._publish_prepared(cached, prepared)
            return prepared


    def _prepare_arenas(
        self,
        arena_bytes: int,
    ) -> None:
        streaming = cast(StreamingSettings, self._streaming)
        buffer_budget = cast(int, self._buffer_budget)
        staging_device = streaming.staging_device
        staging_buffers = streaming.staging_buffers
        if not self._staging_arenas:
            self._staging_device = torch.device(staging_device)
            for index in range(staging_buffers):
                arena = self._allocate_arena(arena_bytes)
                # Retain storage before registration so failed setup can be cleaned up.
                self._staging_arenas.append(arena)
                self._native_setup_started = True
                if staging_device == "cpu":
                    self._staging_registrations.append(
                        self._manager.register_dram_buffer(arena)
                    )
                else:
                    self._manager.register_tensors(
                        {f"__bounded_arena_{index}__": arena}
                    )
        if self._staging_arenas[0].numel() > buffer_budget:
            raise RuntimeError("existing bounded arena exceeds the requested limit")

    def _bind_bounded_workspace(
        self, compiled: _StreamingSchedule, sources: Mapping
    ) -> _BoundedWorkspace:
        bindings = []
        for index, batch in enumerate(compiled.batches):
            arena = self._staging_arenas[index % len(self._staging_arenas)]
            offset = 0
            buffers = []
            for layout in batch.layouts:
                tensors = {}
                for name, (shape, dtype) in layout.items():
                    nbytes = math.prod(shape) * dtype.itemsize
                    tensors[name] = arena[offset:offset + nbytes].view(dtype).view(shape)
                    offset += ((nbytes + 255) // 256) * 256
                buffers.append(tensors)
            recv, convert, full = buffers
            descriptors = tuple(
                _BoundedReadDescriptor(item.session, item.src_addr, item.dst_addr, item.nbytes)
                for item in self._descriptors(batch.transfer_plan, recv, full, convert)
            )
            full_copies = tuple(
                (
                    recv[copy.param_name].as_strided(
                        copy.dest_shape, copy.dest_stride,
                        recv[copy.param_name].storage_offset() + copy.dest_offset,
                    ),
                    full[pull.src_name], copy.op_chain,
                )
                for pull in batch.transfer_plan.full_pulls for copy in pull.copies
            )
            converted = {item.param_name for item in batch.transfer_plan.converts}
            conversion_selections = {
                copy.param_name: copy for copy in batch.capture.copies
                if copy.param_name in converted
            }
            conversion_copies = tuple(
                (
                    recv[copy.param_name].as_strided(
                        copy.dest_shape, copy.dest_stride,
                        recv[copy.param_name].storage_offset() + copy.dest_offset,
                    ),
                    convert[copy.param_name],
                )
                for item in batch.transfer_plan.converts
                for copy in (conversion_selections[item.param_name],)
            )
            bindings.append(_BoundedBatchBinding(
                tuple(buffers), descriptors,
                MappingProxyType({copy.src_name: sources[copy.src_name] for copy in batch.capture.copies}),
                full_copies, conversion_copies,
                tuple(tensor for tensors in buffers for tensor in tensors.values()),
                sum(item.nbytes for item in descriptors),
            ))
        return _BoundedWorkspace(
            compiled, self._workspace_generation,
            tuple((weakref.ref(arena), _arena_geometry(arena)) for arena in self._staging_arenas),
            tuple(bindings), sum(arena.numel() for arena in self._staging_arenas),
        )

    def iter_bounded(
        self, prepared: _PreparedBoundedTransfer, metrics: dict[str, Any]
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield staged batches; callers must commit before advancing.

        With one arena each batch is read, yielded, and committed in turn.
        Content digests are checked only when MX_RESHARD_PUBLISH_DIGEST is
        enabled. With two arenas the READ for batch ``i + 1`` is posted into the
        other arena before batch ``i`` is yielded, allowing asynchronous READs
        to overlap the caller's commit. An arena is only reposted after the
        commit that read from it has been synchronized.
        """
        if self._closed or prepared is not self._active:
            raise RuntimeError("bounded NIXL transfer is no longer active")
        arenas = self._staging_arenas
        assert arenas
        metrics["staging_buffers"] = len(arenas)
        metrics["batches"] = len(prepared.batches)
        batches = prepared.batches
        metrics.update(
            descriptor_cache_hits=0, descriptor_cache_misses=0,
            descriptor_builds=prepared.metrics.get("descriptor_builds", 0)
        )

        plan = self._weight_update_plan
        assert plan is not None and plan.bounded_workspace is not None
        workspace = plan.bounded_workspace
        metrics["staging_peak_bytes"] = workspace.staging_peak_bytes
        reused = bool(prepared.metrics.get("descriptor_binding_reused"))

        def post(index: int) -> tuple:
            if (
                workspace.generation != self._workspace_generation
                or (self._debug_validate_workspace and not workspace.matches(arenas))
            ):
                raise RuntimeError("bounded workspace changed; prepare the transfer again")
            batch = batches[index]
            binding = workspace.batches[index]
            metrics["descriptor_cache_hits" if reused else "descriptor_cache_misses"] += 1
            # Captured loaders may leave padding untouched. Reused arenas must
            # reproduce the zero-filled load layout before the NIC writes it.
            for tensor in binding.zero_tensors:
                tensor.zero_()
            if arenas[index % len(arenas)].device.type == "cuda":
                torch.cuda.synchronize(self._device)
            chunk = _PreparedNixlTransfer(
                batch.transfer_plan, batch.capture, binding.sources,
                binding.descriptors, prepared.transport, checksums=prepared.checksums,
            )
            started = time.perf_counter()
            posted = prepared.transport.post_reads(list(chunk.descriptors))
            return chunk, binding, posted, started

        pending = None
        unwinding = False
        completed = False
        try:
            pending = post(0)
            for index in range(len(batches)):
                chunk, binding, posted, started = pending
                pending = None
                self._recv_buffers, self._convert_buffers, self._full_buffers = binding.buffers
                self._active = chunk
                staged = self._complete_stage(chunk, posted, started, binding=binding)
                if len(arenas) > 1 and index + 1 < len(batches):
                    # The other arena's previous batch was committed and
                    # synchronized one iteration ago, so it is free to refill.
                    pending = post(index + 1)
                for key, value in staged.metrics.items():
                    metrics[key] = metrics.get(key, 0) + value
                yield staged.tensors
                # All installation reads must complete before arena reuse.
                torch.cuda.synchronize(self._device)
                if len(arenas) == 1 and index + 1 < len(batches):
                    pending = post(index + 1)
            completed = True
        except GeneratorExit:
            # Deliberate abandonment, not a failure, so a drain error below has
            # nothing to mask and must still be reported.
            raise
        except BaseException:
            unwinding = True
            raise
        finally:
            if not completed:
                self._invalidate_workspace_binding()
            if pending is not None:
                # A prefetched READ is in flight for a batch the caller will
                # never consume; drain it so the handles are released.
                try:
                    prepared.transport.await_reads(pending[2])
                except Exception as error:
                    # Not merely uncleaned: until this READ is drained the arena
                    # may still receive RDMA writes, so reusing or freeing it is
                    # unsafe. Report it rather than returning as if the transfer
                    # had ended, and only downgrade to a log when an earlier
                    # failure is already propagating and must not be masked.
                    logger.error(
                        "draining a prefetched bounded READ batch failed; the "
                        "staging arena may still be written by an in-flight read "
                        "and the generator engine must be restarted",
                        exc_info=True,
                    )
                    if not unwinding:
                        raise RuntimeError(
                            "a prefetched bounded READ could not be drained, so the "
                            "staging arena may still be written; restart the "
                            "generator engine"
                        ) from error
            self._active = prepared

    @staticmethod
    def _validate_complete(
        capture: CaptureResult,
        parameter_layout: dict[str, tuple[tuple[int, ...], torch.dtype]],
        plan: TransferPlan,
    ) -> None:
        written = {copy.param_name for copy in capture.copies}
        missing = sorted(set(parameter_layout) - written)
        unsupported = list(capture.unsupported)
        if missing or unsupported or capture.unattributed or plan.fallback:
            causes = summarize_unsupported(capture.unsupported_reasons)
            raise IncompleteRefit(
                "full-tensor refit must cover every engine parameter; "
                f"missing={len(missing)}, unsupported={len(unsupported)}, "
                f"unattributed={capture.unattributed}, fallback={len(plan.fallback)}, "
                f"causes={causes}, missing_names={missing[:10]}"
            )

    @staticmethod
    def _layout(tensors: dict[str, torch.Tensor]) -> dict:
        return {
            name: (tuple(tensor.shape), tensor.dtype)
            for name, tensor in tensors.items()
        }

    def _ensure_buffers(
        self,
        current: dict[str, torch.Tensor],
        expected: dict[str, tuple[tuple[int, ...], torch.dtype]],
        *,
        label: str,
    ) -> None:
        if current:
            if self._layout(current) != expected:
                raise RuntimeError(
                    f"{label} layout changed; restart the generator engine"
                )
            return
        with classic_cuda_alloc():
            current.update(
                {
                    name: torch.empty(shape, dtype=dtype, device=self._device)
                    for name, (shape, dtype) in expected.items()
                }
            )

    def _ensure_workspace(
        self,
        plan: TransferPlan,
        parameter_layout: dict[str, tuple[tuple[int, ...], torch.dtype]],
    ) -> None:
        recv_expected = {
            name: (tuple(shape), dtype)
            for name, (shape, dtype) in parameter_layout.items()
        }
        self._ensure_buffers(self._recv_buffers, recv_expected, label="receive-buffer")

        convert_expected = {
            convert.param_name: (tuple(convert.dest_shape), convert.src_dtype)
            for convert in plan.converts
        }
        self._ensure_buffers(
            self._convert_buffers, convert_expected, label="conversion-buffer"
        )
        full_expected = {
            full.src_name: (tuple(full.global_shape), full.dtype)
            for full in plan.full_pulls
        }
        self._ensure_buffers(
            self._full_buffers, full_expected, label="full-pull buffer"
        )

        recv_params = set(recv_expected)
        if self._registered_recv_params and self._registered_recv_params != recv_params:
            raise RuntimeError(
                "receive parameter set changed; restart the generator engine"
            )
        if convert_expected and not self._convert_registered:
            self._native_setup_started = True
            self._manager.register_tensors(
                {
                    f"__convert__{name}": tensor
                    for name, tensor in self._convert_buffers.items()
                }
            )
            self._convert_registered = True
        if full_expected and not self._full_registered:
            self._native_setup_started = True
            self._manager.register_tensors(
                {
                    f"__full__{name}": tensor
                    for name, tensor in self._full_buffers.items()
                }
            )
            self._full_registered = True
        if not self._registered_recv_params and recv_params:
            self._native_setup_started = True
            self._manager.register_tensors(self._recv_buffers)
            self._registered_recv_params = recv_params

    def _descriptors(
        self,
        plan: TransferPlan,
        recv: dict[str, torch.Tensor] | None = None,
        full: dict[str, torch.Tensor] | None = None,
        convert: dict[str, torch.Tensor] | None = None,
    ) -> list[ReadDescriptor]:
        recv_buffers = self._recv_buffers if recv is None else recv
        full_buffers = self._full_buffers if full is None else full
        convert_buffers = self._convert_buffers if convert is None else convert
        descriptors = exact_descriptors(
            plan, lambda name: recv_buffers[name].data_ptr()
        )
        descriptors.extend(
            ReadDescriptor(
                session=segment.session,
                src_addr=segment.src_addr,
                dst_addr=full_buffers[pull.src_name].data_ptr() + segment.dst_byte,
                nbytes=segment.nbytes,
            )
            for pull in plan.full_pulls
            for segment in pull.segments
        )
        descriptors.extend(
            ReadDescriptor(
                session=segment.session,
                src_addr=segment.src_addr,
                dst_addr=convert_buffers[conv.param_name].data_ptr() + segment.dst_byte,
                nbytes=segment.nbytes,
            )
            for conv in plan.converts
            for segment in conv.segments
        )
        return descriptors

    def stage(self, prepared: _PreparedNixlTransfer) -> _StagedNixlWeights:
        """Pull, reconstruct, convert, and verify without touching live weights."""
        if self._closed:
            raise RuntimeError("NIXL staged transfer is closed")
        if prepared is not self._active:
            raise RuntimeError("NIXL transfer plan is no longer active")
        started = time.perf_counter()
        posted = prepared.transport.post_reads(list(prepared.descriptors))
        return self._complete_stage(prepared, posted, started)

    @torch.no_grad()
    def _complete_stage(
        self, prepared: _PreparedNixlTransfer, posted: list, started: float,
        *, binding: _BoundedBatchBinding | None = None,
    ) -> _StagedNixlWeights:
        """Wait for posted READs, then reconstruct, convert, and verify."""
        wait_started = time.perf_counter()
        prepared.transport.await_reads(posted)
        wire_wait_seconds = time.perf_counter() - wait_started
        wire_seconds = time.perf_counter() - started

        reconstruct_started = time.perf_counter()
        if binding is not None:
            for destination, source, operations in binding.full_copies:
                destination.copy_(_replay_ops(source, operations))
            for destination, source in binding.conversion_copies:
                destination.copy_(source)
        else:
            for full in prepared.transfer_plan.full_pulls:
                source = self._full_buffers[full.src_name]
                for copy in full.copies:
                    destination = self._recv_buffers[copy.param_name].as_strided(
                        copy.dest_shape,
                        copy.dest_stride,
                        self._recv_buffers[copy.param_name].storage_offset()
                        + copy.dest_offset,
                    )
                    destination.copy_(_replay_ops(source, copy.op_chain))
            converted = {convert.param_name for convert in prepared.transfer_plan.converts}
            conversion_copies = {
                copy.param_name: copy
                for copy in prepared.capture.copies
                if copy.param_name in converted
            }
            for convert in prepared.transfer_plan.converts:
                copy = conversion_copies[convert.param_name]
                target = self._recv_buffers[convert.param_name]
                destination = target.as_strided(
                    copy.dest_shape,
                    copy.dest_stride,
                    target.storage_offset() + copy.dest_offset,
                )
                destination.copy_(self._convert_buffers[convert.param_name])
        torch.cuda.synchronize(self._device)
        reconstruct_seconds = time.perf_counter() - reconstruct_started
        # Only digest mode has complete tensors and stamped digests to check.
        if envs.MX_RESHARD_PUBLISH_DIGEST:
            self._verify(prepared)

        bytes_received = binding.wire_bytes if binding is not None else sum(d.nbytes for d in prepared.descriptors)
        # This is the path the FSDP trainer refits over, and the path the 20x
        # collapse was measured on, so it is the one the floor most needs to cover.
        throughput.warn_if_below_floor(
            wire_bytes=bytes_received,
            wire_seconds=wire_seconds,
            log=logger,
            context={"device_id": self._device_id, "phase": "stage"},
        )
        logger.info(
            "[TIMING] staged xfer: %.3f GB, %d descriptors "
            "(seg=%d full_pull=%d convert=%d), %d tensors | "
            "wire=%.3fs reconstruct=%.3fs",
            bytes_received / 1e9,
            len(prepared.descriptors),
            len(prepared.transfer_plan.segments),
            len(prepared.transfer_plan.full_pulls),
            len(prepared.transfer_plan.converts),
            len(self._recv_buffers),
            wire_seconds,
            reconstruct_seconds,
        )
        return _StagedNixlWeights(
            tensors=self._recv_buffers,
            metrics={
                "bytes_received": bytes_received,
                "segments": len(prepared.descriptors),
                "wire_s": wire_seconds,
                "wire_wait_s": wire_wait_seconds,
                "reconstruct_s": reconstruct_seconds,
                "full_pull_sources": len(prepared.transfer_plan.full_pulls),
                "converts": len(prepared.transfer_plan.converts),
            },
        )

    @staticmethod
    def _validate_peer_manifest(
        manifest: p2p_pb2.GetTensorManifestResponse,
        destination_tensors: dict[str, torch.Tensor],
    ) -> list[TensorDescriptor]:
        source_tensors = [
            TensorDescriptor(
                name=tensor.name,
                addr=tensor.addr,
                size=tensor.size,
                device_id=tensor.device_id,
                dtype=tensor.dtype,
            )
            for tensor in manifest.tensors
        ]
        if not source_tensors:
            raise RuntimeError("P2P source has no tensor descriptors")

        source_names = {tensor.name for tensor in source_tensors}
        local_names = set(destination_tensors)
        if source_names != local_names:
            local_only = sorted(local_names - source_names)
            source_only = sorted(source_names - local_names)
            raise ManifestMismatchError(
                "runtime tensor name mismatch: "
                f"{len(local_only)} local-only (first: {local_only[:5]}), "
                f"{len(source_only)} source-only (first: {source_only[:5]})"
            )
        for source in source_tensors:
            destination = destination_tensors[source.name]
            size = destination.numel() * destination.element_size()
            if source.size != size or source.dtype != str(destination.dtype):
                raise ManifestMismatchError(
                    f"runtime tensor metadata mismatch for {source.name!r}"
                )
        return source_tensors

    @staticmethod
    def _peer_endpoint(
        manifest: p2p_pb2.GetTensorManifestResponse,
    ) -> tuple[str, int, str]:
        endpoint = manifest.metadata_endpoint
        try:
            host, port_text = endpoint.rsplit(":", 1)
            port = int(port_text)
        except ValueError as error:
            raise RuntimeError(
                f"P2P source published an unusable metadata endpoint: {endpoint!r}"
            ) from error
        if not host or not 1 <= port <= 65535 or not manifest.agent_name:
            raise RuntimeError(
                "P2P source published unusable NIXL connection metadata: "
                f"endpoint={endpoint!r}, agent_name={manifest.agent_name!r}"
            )
        return host, port, manifest.agent_name

    def prepare_peer_read(
        self,
        *,
        source: p2p_pb2.WorkerMetadata,
        mx_source_id: str,
        worker_id: str,
        destination_tensors: dict[str, torch.Tensor],
    ) -> TensorReadLease:
        """Validate and retain a donor lease until apply or release."""
        if self._closed:
            raise RuntimeError("NIXL staged transfer is closed")
        if not source.worker_grpc_endpoint:
            raise RuntimeError("generator P2P source has no tensor lease endpoint")
        lease, _ = prepare_tensor_read(
            source.worker_grpc_endpoint,
            mx_source_id,
            worker_id=worker_id,
            timeout=self._timeout,
        )
        try:
            self._validate_peer_manifest(lease.manifest, destination_tensors)
            self._peer_endpoint(lease.manifest)
        except BaseException:
            lease.close()
            raise
        return lease

    def receive_peer(
        self,
        *,
        tensor_read: TensorReadLease,
        destination_tensors: dict[str, torch.Tensor],
        on_transfer_start: Callable[[], None],
    ) -> dict[str, Any]:
        """Pull a peer's runtime tensors directly into live engine storage."""
        if self._closed:
            raise RuntimeError("NIXL staged transfer is closed")

        remote_agent_name: str | None = None
        started = time.perf_counter()
        manifest = tensor_read.manifest
        source_tensors = self._validate_peer_manifest(
            manifest,
            destination_tensors,
        )
        host, port, remote_agent_name = self._peer_endpoint(manifest)
        try:
            self._manager.fetch_remote_and_wait(
                remote_agent_name=remote_agent_name,
                ip=host,
                port=port,
                timeout_seconds=self._timeout,
            )
            bytes_received, tensor_count, wire_seconds = (
                self._manager.receive_from_source(
                    source_metadata=b"",
                    source_tensors=source_tensors,
                    timeout_seconds=self._timeout,
                    remote_agent_name=remote_agent_name,
                    require_exact_match=True,
                    destination_tensors=destination_tensors,
                    on_transfer_start=on_transfer_start,
                )
            )
        finally:
            if remote_agent_name is not None:
                self._manager.remove_remote_agent(remote_agent_name)

        throughput.warn_if_below_floor(
            wire_bytes=bytes_received,
            wire_seconds=wire_seconds,
            log=logger,
            context={"device_id": self._device_id, "phase": "receive_peer"},
        )
        return {
            "bytes_received": bytes_received,
            "segments": tensor_count,
            "wire_s": round(wire_seconds, 6),
            "peer_s": round(time.perf_counter() - started, 6),
        }

    def _verification_tensor(self, prepared: _PreparedNixlTransfer, name: str):
        source = prepared.sources[name]
        if name in self._full_buffers:
            return self._full_buffers[name]
        copy = next(
            (
                copy
                for copy in prepared.capture.copies
                if copy.src_name == name
                and not copy.op_chain
                and tuple(copy.dest_shape) == tuple(source.global_shape)
            ),
            None,
        )
        if copy is None:
            raise RuntimeError(f"cannot recover complete staged source {name!r}")
        if copy.param_name in self._convert_buffers:
            return self._convert_buffers[copy.param_name]
        buffer = self._recv_buffers[copy.param_name]
        return buffer.as_strided(
            copy.dest_shape,
            copy.dest_stride,
            buffer.storage_offset() + copy.dest_offset,
        )

    def _verify(self, prepared: _PreparedNixlTransfer) -> None:
        for name, source in prepared.sources.items():
            tensor = self._verification_tensor(prepared, name)
            for shard in source.shards:
                expected = prepared.checksums.get(
                    (
                        name, shard.session, shard.addr,
                        tuple(shard.shard_offset), tuple(shard.shape),
                    )
                )
                if not expected:
                    raise RuntimeError(
                        f"source {name!r} did not publish a verification digest"
                    )
                actual = tensor_digest(
                    shard_region(
                        tensor,
                        source.global_shape,
                        shard.shard_offset,
                        shard.shape,
                    )
                )
                if actual != expected:
                    raise RuntimeError(
                        f"staged weight digest mismatch for source {name!r} "
                        f"at offset {tuple(shard.shard_offset)}"
                    )

    def close(self) -> None:
        if self._closed:
            return
        self._invalidate_workspace_binding()
        self._weight_update_plan = None
        self._full_copy_descriptors = None
        self._closed = True
        if self._owns_manager:
            self._release_staging_registrations()
            self._manager.shutdown()
            self._active = None
            # The agent's registrations are gone, so staging storage can be
            # freed eagerly. A shared agent may still hold these buffers
            # registered; keep them referenced until its owner shuts down.
            self._recv_buffers.clear()
            self._convert_buffers.clear()
            self._full_buffers.clear()
            self._staging_arenas.clear()
            self._staging_device = None


__all__: list[str] = []

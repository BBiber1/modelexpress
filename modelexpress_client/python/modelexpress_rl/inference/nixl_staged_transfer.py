# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-version staged NIXL transfer for RL generator workers.

This module owns the state that makes a pull transfer correct: the selected
source manifests, the physical plan, registered destination buffers, peer
metadata, transfer completion, and verification. It deliberately does not know
how an inference engine captures its load layout or installs received weights.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from types import MappingProxyType
from typing import Any, NamedTuple

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
from modelexpress.refit.reshard.slice_plan import plan_pull
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

from modelexpress_rl.inference._cache_config import RefitCacheConfig
from modelexpress_rl.inference._source_snapshot import (
    _freeze_sources,
    _source_structure,
)
from modelexpress_rl.inference.plan import TrainerSourceSnapshot

# Named under modelexpress.* (not modelexpress_rl) so the per-update summary surfaces
# in the vLLM engine process, which only configures the modelexpress logger.
logger = logging.getLogger("modelexpress.reshard.staged_transfer")


@dataclass(frozen=True)
class _ResolvedSources:
    sources: dict
    session_to_agent: dict
    session_to_device: dict
    agent_metadata: dict[str, bytes]
    session_to_memory: dict[str, str] = field(default_factory=dict)


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


@dataclass(frozen=True)
class _CachedPullPlan:
    trainer_source_snapshot: TrainerSourceSnapshot
    generator_capture_snapshot: CaptureResult
    parameter_layout: MappingProxyType
    compiled: TransferPlan | _CompiledBoundedPlan
    manifests: tuple[bytes, ...]
    required_agents: frozenset[str] = frozenset()
    used_sources: MappingProxyType | None = None


@dataclass(frozen=True)
class _PreparedNixlTransfer:
    """One immutable physical plan over reusable registered destinations."""

    plan: TransferPlan
    capture: CaptureResult
    sources: dict
    descriptors: tuple[ReadDescriptor | _BoundedReadDescriptor, ...]
    transport: NixlReshardTransport
    metrics: dict[str, float] = field(default_factory=dict)


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
class _BoundedBatch:
    capture: CaptureResult
    plan: TransferPlan
    layouts: _StagingLayouts
    nbytes: int

    def __post_init__(self) -> None:
        # Readers reach these views both by name and by position, and
        # dataclasses.replace or a plain 3-tuple would satisfy only the second.
        # Coerce so the annotation holds however the batch was built.
        if not isinstance(self.layouts, _StagingLayouts):
            object.__setattr__(self, "layouts", _StagingLayouts(*self.layouts))


@dataclass(frozen=True)
class _PreparedBoundedTransfer:
    batches: tuple[_BoundedBatch, ...]
    sources: dict
    transport: NixlReshardTransport
    metrics: dict[str, float] = field(default_factory=dict)


def _require_positive_bytes(value: object, name: str) -> int:
    """Return ``value`` as a byte count, rejecting bool and non-positive ints."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class _CompiledBoundedPlan:
    plan: TransferPlan
    module_batches: tuple[_BoundedBatch, ...]
    batches: tuple[_BoundedBatch, ...]
    required_agents: frozenset[str]


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
class _BoundedDescriptors:
    """One validated plan, weak arena identities, and no transfer resources."""

    plan: _CompiledBoundedPlan
    generation: int
    arenas: tuple[tuple[weakref.ReferenceType, tuple], ...]
    batches: tuple[tuple[_BoundedReadDescriptor, ...] | None, ...]

    def matches(
        self,
        batches: tuple[_BoundedBatch, ...],
        generation: int,
        arenas: list[torch.Tensor],
    ) -> bool:
        return (
            self.plan.batches is batches
            and self.generation == generation
            and len(self.arenas) == len(arenas)
            and all(
                type(arena) is torch.Tensor
                and reference() is arena
                and geometry == _arena_geometry(arena)
                for (reference, geometry), arena in zip(self.arenas, arenas)
            )
        )


def _compile_bounded_plan(
    *,
    resolved: _ResolvedSources,
    capture: CaptureResult,
    parameter_layout: _StagingLayout,
    max_staging_bytes: int,
    metrics: dict[str, Any],
    staging_buffers: int = 1,
    total_staging_bytes: int | None = None,
    cache_config: RefitCacheConfig,
) -> _CompiledBoundedPlan:
    metrics.update(
        plan_cache_enabled=int(cache_config.cache_plan),
        plan_cache_hits=0,
        plan_cache_misses=1,
        bounded_whole_plan_builds=0,
        owner_plan_builds=0,
    )
    started = time.perf_counter()
    plan = _plan_staged_transfer(
        capture, resolved.sources, publish_digest=cache_config.publish_digest
    )
    metrics["initial_whole_plan_s"] = time.perf_counter() - started
    modules = _bounded_batches(
        capture,
        parameter_layout,
        resolved.sources,
        max_staging_bytes,
        complete_plan=plan,
        metrics=metrics,
        total_staging_bytes=total_staging_bytes,
        staging_buffers=staging_buffers,
        publish_digest=cache_config.publish_digest,
    )
    batches = (
        _pack_bounded_batches(modules, max_staging_bytes)
        if cache_config.pack_modules
        else modules
    )
    required_agents = frozenset(
        agent
        for transfer_plan in (plan, *(batch.plan for batch in batches))
        for agent in _required_agent_metadata(transfer_plan, resolved)
    )
    return _CompiledBoundedPlan(plan, modules, batches, required_agents)


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
    publish_digest: bool | None = None,
) -> tuple[_BoundedBatch, ...]:
    """Validate all owning-module batches before allocating or installing."""
    if metrics is None:
        metrics = {}
    started = time.perf_counter()
    complete = complete_plan
    if complete is None:
        complete = _plan_staged_transfer(
            capture, sources, publish_digest=publish_digest
        )
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
        plan = _plan_staged_transfer(subset, sources, publish_digest=publish_digest)
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
        batches.append(_BoundedBatch(subset, plan, layouts, nbytes))
    if not batches:
        raise IncompleteRefit("bounded refit has no engine parameters")
    return tuple(batches)


def _pack_bounded_batches(
    batches: tuple[_BoundedBatch, ...], max_staging_bytes: int
) -> tuple[_BoundedBatch, ...]:
    """Pack complete modules without changing source READ ranges.

    Owning-module batches are the unit of correctness; this only coalesces
    neighbours that fit the same arena together, so each packed batch still
    installs whole modules and reads exactly the bytes the unpacked plan read.
    """

    def merge(group: list[_BoundedBatch]) -> _BoundedBatch:
        capture = CaptureResult(
            copies=[copy for batch in group for copy in batch.capture.copies]
        )
        plan = TransferPlan()
        layouts = _StagingLayouts({}, {}, {})
        for batch in group:
            _merge_plan(plan, batch.plan)
            for layout, incoming in zip(layouts, batch.layouts, strict=True):
                layout.update(incoming)
        return _BoundedBatch(
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
    capture: CaptureResult, sources: dict, *, publish_digest: bool | None = None
) -> TransferPlan:
    """Plan reads for each source.

    Default: minimal slice reads via plan_transfer (a partial read of a shard cannot
    be whole-shard digest-verified, so correctness rests on the coverage gate). Under
    MX_RESHARD_PUBLISH_DIGEST (verification mode): reconstruct every source that isn't
    a whole-tensor identity copy as a full pull, so _verify has a complete shard to
    digest-check.
    """
    if publish_digest is None:
        publish_digest = envs.MX_RESHARD_PUBLISH_DIGEST
    if not publish_digest:
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
        verify_trainer_mesh: Callable[[TrainerSourceSnapshot], None] | None = None,
        cache_config: RefitCacheConfig | None = None,
    ) -> None:
        self.cache_config = cache_config or RefitCacheConfig()
        self._generator_layout: tuple[tuple, CaptureResult, MappingProxyType] | None = (
            None
        )
        self._trainer_snapshot: TrainerSourceSnapshot | None = None
        self._verify_trainer_mesh = verify_trainer_mesh
        self._transport: NixlReshardTransport | None = None
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
        self._loaded_agent_metadata: dict[str, bytes] = {}
        self._closed = False
        # Bounded staging: one or two byte arenas on CUDA or pinned host memory.
        # Host arenas are registered as NIXL DRAM and tracked here so they can be
        # deregistered before the agent shuts down.
        self._staging_arenas: list[torch.Tensor] = []
        self._staging_registrations: list[Any] = []
        self._staging_device: torch.device | None = None
        self._workspace_mode: str | None = None
        self._cached_pull_plan: _CachedPullPlan | None = None
        self._full_copy_descriptors: tuple[ReadDescriptor, ...] | None = None
        self._workspace_generation = 0
        self._descriptor_cache: _BoundedDescriptors | None = None

    @property
    def _bounded_arena(self) -> torch.Tensor | None:
        return self._staging_arenas[0] if self._staging_arenas else None

    @_bounded_arena.setter
    def _bounded_arena(self, value: torch.Tensor | None) -> None:
        self._invalidate_descriptors()
        self._staging_arenas = [] if value is None else [value]

    def _invalidate_descriptors(self) -> None:
        self._descriptor_cache = None
        self._workspace_generation += 1

    def _release_staging_registrations(self) -> None:
        self._invalidate_descriptors()
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
        self._invalidate_descriptors()
        if not self._owns_manager:
            # A shared agent still serves its owner; tearing it down here would
            # deregister memory we do not own. Only a transfer-owned agent can
            # be cycled to release staging storage safely.
            raise RuntimeError(
                "resetting staging storage requires a transfer-owned NIXL agent; "
                "restart the generator engine"
            )
        if self._workspace_mode is None:
            return
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
        self._workspace_mode = None
        self._cached_pull_plan = None
        self._full_copy_descriptors = None
        self._transport = None

    def _select_workspace_mode(self, mode: str) -> None:
        """Replace released staging storage only after tearing down its agent."""
        if self._workspace_mode == mode:
            return
        if self._workspace_mode is not None:
            self.reset_workspace()
        if self._owns_manager:
            # The constructor initialized an owned agent and reset_workspace()
            # shuts it down, so a mode change has to bring it back. A borrowed
            # agent is initialized and torn down by its owner, and
            # reset_workspace() refuses to cycle one, so it needs neither.
            try:
                self._manager.initialize()
            except Exception:
                self._manager.shutdown()
                raise
        self._workspace_mode = mode

    def cached_trainer_source(self) -> TrainerSourceSnapshot | None:
        return None if self._closed else self._trainer_snapshot

    def _reuse_mesh_plan(
        self,
        previous: _CachedPullPlan | None,
        source: TrainerSourceSnapshot,
        manifests: list[bytes] | None,
        capture_layout: _CaptureLayout,
        metrics: dict[str, Any],
    ) -> _CachedPullPlan | None:
        if previous is None:
            return None
        if not (
            self.cache_config.cache_plan and self.cache_config.cache_generator_layout
        ):
            return None
        cached = previous.trainer_source_snapshot
        diagnostic = self.cache_config.validate_plan or self.cache_config.publish_digest
        if source is not cached:
            if (source.mesh_id, source.mesh_generation) != (
                cached.mesh_id,
                cached.mesh_generation,
            ):
                return None
            if not diagnostic:
                return None
            if source.physical_fingerprint != cached.physical_fingerprint:
                if self.cache_config.validate_plan:
                    raise ValueError(
                        "trainer source structure changed under unchanged mesh identity"
                    )
                return None
        metrics.update(
            plan_cache_enabled=1,
            plan_cache_hits=1,
            plan_cache_misses=0,
            owner_plan_builds=0,
            bounded_whole_plan_builds=0,
            source_cache_enabled=1,
            source_cache_hits=1,
            source_cache_misses=0,
            source_manifest_bytes=0,
        )
        if not (diagnostic or self.cache_config.validate_generator_layout):
            return previous
        trainer = cached
        if diagnostic:
            resolved = self._resolve_metadata(manifests, metrics)
            old = cached.resolved_metadata
            if (
                resolved.session_to_agent != old.session_to_agent
                or resolved.session_to_device != old.session_to_device
                or resolved.session_to_memory != old.session_to_memory
                or resolved.agent_metadata != old.agent_metadata
                or set(resolved.sources) != set(old.sources)
                or any(
                    _source_structure(row) != _source_structure(old.sources[name])
                    for name, row in resolved.sources.items()
                )
            ):
                raise ValueError(
                    "trainer source metadata changed under unchanged mesh identity"
                )
            trainer = replace(
                source,
                resolved_metadata=resolved,
            )
        capture = previous.generator_capture_snapshot
        parameter_layout = previous.parameter_layout
        if self.cache_config.validate_generator_layout:
            resolved = trainer.resolved_metadata
            manifest = [
                (name, row.dtype, tuple(row.global_shape))
                for name, row in resolved.sources.items()
            ]
            current_capture, current_layout = capture_layout(manifest)
            if not _layout_values_match(
                (capture, dict(parameter_layout)),
                (current_capture, current_layout),
            ):
                raise ValueError("generator layout changed while reusing cached plan")
        if self.cache_config.validate_plan:
            compiled = previous.compiled
            plan = (
                compiled.plan
                if isinstance(compiled, _CompiledBoundedPlan)
                else compiled
            )
            self._validate_complete(capture, parameter_layout, plan)
            if isinstance(compiled, _CompiledBoundedPlan):
                for batch in compiled.module_batches:
                    self._validate_complete(
                        batch.capture, batch.layouts.recv, batch.plan
                    )
        used_sources = previous.used_sources
        if diagnostic and used_sources is not None:
            used_sources = MappingProxyType(
                {name: trainer.resolved_metadata.sources[name] for name in used_sources}
            )
        return replace(
            previous,
            trainer_source_snapshot=trainer,
            manifests=tuple(manifests) if manifests is not None else previous.manifests,
            used_sources=used_sources,
        )

    def _verify_completed_plan(
        self, trainer: TrainerSourceSnapshot, *, reusable: bool
    ) -> None:
        if not reusable and self._verify_trainer_mesh is not None:
            self._verify_trainer_mesh(trainer)

    def _resolve_metadata(
        self, manifests: list[bytes], metrics: dict[str, Any]
    ) -> _ResolvedSources:
        metrics.update(
            source_cache_enabled=int(self.cache_config.cache_plan),
            source_cache_hits=0,
            source_cache_misses=1,
            source_manifest_bytes=sum(len(blob) for blob in manifests),
        )
        resolved = _resolve_sources(manifests, metrics=metrics)
        return replace(
            resolved,
            sources=_freeze_sources(resolved.sources),
            session_to_agent=MappingProxyType(dict(resolved.session_to_agent)),
            session_to_device=MappingProxyType(dict(resolved.session_to_device)),
            session_to_memory=MappingProxyType(dict(resolved.session_to_memory)),
            agent_metadata=MappingProxyType(dict(resolved.agent_metadata)),
        )

    def _resolve_layout(
        self,
        manifests: list[bytes],
        capture_layout: _CaptureLayout,
        metrics: dict[str, Any],
        trainer_snapshot: TrainerSourceSnapshot,
    ) -> tuple[TrainerSourceSnapshot, CaptureResult, MappingProxyType]:
        started = time.perf_counter()
        resolved = trainer_snapshot.resolved_metadata
        if (
            resolved is None
            or not self.cache_config.cache_plan
            or self.cache_config.validate_plan
            or self.cache_config.publish_digest
        ):
            resolved = self._resolve_metadata(manifests, metrics)
        else:
            metrics.update(
                source_cache_hits=1, source_cache_misses=0, source_manifest_bytes=0
            )
        trainer = (
            trainer_snapshot
            if resolved is trainer_snapshot.resolved_metadata
            else replace(trainer_snapshot, resolved_metadata=resolved)
        )
        manifest = tuple(
            (name, source.dtype, tuple(source.global_shape))
            for name, source in resolved.sources.items()
        )
        metrics["source_metadata_s"] = time.perf_counter() - started
        retained = self._generator_layout
        if (
            self.cache_config.cache_generator_layout
            and retained is not None
            and retained[0] == manifest
        ):
            if self.cache_config.validate_generator_layout:
                current = capture_layout(list(manifest))
                if not _layout_values_match((retained[1], dict(retained[2])), current):
                    raise ValueError(
                        "generator layout changed while reusing cached layout"
                    )
            metrics["layout_capture_s"] = 0.0
            return trainer, retained[1], retained[2]
        started = time.perf_counter()
        capture, parameter_layout = deepcopy(capture_layout(list(manifest)))
        parameter_layout = MappingProxyType(parameter_layout)
        self._generator_layout = (
            (manifest, capture, parameter_layout)
            if self.cache_config.cache_generator_layout
            else None
        )
        metrics["layout_capture_s"] = time.perf_counter() - started
        return trainer, capture, parameter_layout

    def _connect_sources(
        self,
        resolved: _ResolvedSources,
        required_metadata: dict[str, bytes],
        *,
        host_staging: bool = False,
    ) -> NixlReshardTransport:
        changed = {
            agent: metadata
            for agent, metadata in required_metadata.items()
            if self._loaded_agent_metadata.get(agent) != metadata
        }
        obsolete = [
            agent
            for agent, metadata in self._loaded_agent_metadata.items()
            if required_metadata.get(agent) != metadata
        ]
        if obsolete and not self._owns_manager:
            raise RuntimeError(
                "cannot replace source agents on a borrowed NIXL manager"
            )
        for agent in obsolete:
            if not self._manager.remove_remote_agent(agent):
                mode = self._workspace_mode
                self.reset_workspace()
                assert mode is not None
                self._select_workspace_mode(mode)
                changed = required_metadata
                break
            del self._loaded_agent_metadata[agent]
        _load_agent_metadata(self._manager, changed)
        self._loaded_agent_metadata.update(changed)
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
        previous_layout = self._generator_layout
        try:
            yield
        except Exception:
            self._generator_layout = previous_layout
            self._trainer_snapshot = None
            self._cached_pull_plan = None
            self._full_copy_descriptors = None
            self._transport = None
            raise

    def _publish_prepared(
        self,
        cached: _CachedPullPlan,
        prepared: _PreparedNixlTransfer | _PreparedBoundedTransfer,
        *,
        reusable: bool,
    ) -> None:
        self._verify_completed_plan(cached.trainer_source_snapshot, reusable=reusable)
        self._transport = prepared.transport
        self._cached_pull_plan = cached
        self._trainer_snapshot = cached.trainer_source_snapshot
        self._active = prepared

    def prepare_full_copy(
        self,
        *,
        manifests: list[bytes] | None,
        capture_layout: _CaptureLayout,
        trainer_snapshot: TrainerSourceSnapshot,
    ) -> _PreparedNixlTransfer:
        """Create a version-scoped transfer over one reusable full-copy plan."""
        with self._preparing():
            self._descriptor_cache = None
            metrics: dict[str, Any] = {}
            warm = self._reuse_mesh_plan(
                self._cached_pull_plan,
                trainer_snapshot,
                manifests,
                capture_layout,
                metrics,
            )
            reusable = warm is not None and isinstance(warm.compiled, TransferPlan)
            if reusable:
                trainer = warm.trainer_source_snapshot
                capture = warm.generator_capture_snapshot
                parameter_layout = warm.parameter_layout
                plan = warm.compiled
                transport = self._transport
                required_agents = warm.required_agents
                used_sources = warm.used_sources
            else:
                self._cached_pull_plan = None
                self._select_workspace_mode("full")
                if manifests is None:
                    manifests = [row.manifest for row in trainer_snapshot.shards]
                trainer, capture, parameter_layout = self._resolve_layout(
                    manifests, capture_layout, metrics, trainer_snapshot
                )
                started = time.perf_counter()
                plan = _plan_staged_transfer(
                    capture,
                    trainer.resolved_metadata.sources,
                    publish_digest=self.cache_config.publish_digest,
                )
                metrics.update(
                    plan_cache_hits=0, plan_cache_misses=1, owner_plan_builds=1
                )
                metrics["initial_whole_plan_s"] = time.perf_counter() - started
                validation_started = time.perf_counter()
                self._validate_complete(capture, parameter_layout, plan)
                metrics["initial_whole_validation_s"] = (
                    time.perf_counter() - validation_started
                )
                metrics["transfer_planning_s"] = time.perf_counter() - started
                resolved = trainer.resolved_metadata
                required_metadata = _required_agent_metadata(plan, resolved)
                required_agents = frozenset(required_metadata)
                expected = _StagingLayouts(
                    dict(parameter_layout),
                    {
                        item.param_name: (tuple(item.dest_shape), item.src_dtype)
                        for item in plan.converts
                    },
                    {
                        item.src_name: (tuple(item.global_shape), item.dtype)
                        for item in plan.full_pulls
                    },
                )
                self._reset_incompatible_full_workspace(expected)
                transport = self._connect_sources(resolved, required_metadata)
                self._ensure_workspace(expected)
                self._full_copy_descriptors = tuple(self._descriptors(plan))
                used_sources = MappingProxyType(
                    {
                        copy.src_name: resolved.sources[copy.src_name]
                        for copy in capture.copies
                    }
                )
            assert transport is not None
            prepared = _PreparedNixlTransfer(
                plan,
                capture,
                used_sources,
                self._full_copy_descriptors,
                transport,
                metrics,
            )
            cached = (
                warm
                if reusable
                else _CachedPullPlan(
                    trainer,
                    capture,
                    MappingProxyType(parameter_layout),
                    plan,
                    tuple(manifests) if manifests is not None else warm.manifests,
                    required_agents,
                    used_sources,
                )
            )
            self._publish_prepared(cached, prepared, reusable=reusable)
            return prepared

    def prepare_streaming(
        self,
        *,
        manifests: list[bytes] | None,
        capture_layout: _CaptureLayout,
        max_staging_bytes: int,
        staging_device: str = "cuda",
        staging_buffers: int = 1,
        trainer_snapshot: TrainerSourceSnapshot,
    ) -> _PreparedBoundedTransfer:
        """Create fresh deferred reads over one reusable bounded pull plan."""
        with self._preparing():
            previous_descriptors, self._descriptor_cache = self._descriptor_cache, None
            previous = self._cached_pull_plan
            metrics = {}
            warm = self._reuse_mesh_plan(
                previous, trainer_snapshot, manifests, capture_layout, metrics
            )
            if warm is not None and isinstance(warm.compiled, _CompiledBoundedPlan):
                assert self._transport is not None
                compiled = warm.compiled
                prepared = _PreparedBoundedTransfer(
                    compiled.batches,
                    warm.trainer_source_snapshot.resolved_metadata.sources,
                    self._transport,
                    metrics,
                )
                self._descriptor_cache = self._prepare_bounded_descriptors(
                    compiled, previous_descriptors, enabled=True
                )
                self._publish_prepared(warm, prepared, reusable=True)
                return prepared
            buffer_budget = self._validate_streaming_settings(
                max_staging_bytes, staging_device, staging_buffers
            )
            self._cached_pull_plan = None
            self._select_workspace_mode(f"bounded:{staging_device}:{staging_buffers}")
            if manifests is None:
                manifests = [row.manifest for row in trainer_snapshot.shards]
            trainer, capture, parameter_layout = self._resolve_layout(
                manifests, capture_layout, metrics, trainer_snapshot
            )
            self._cached_pull_plan = None
            started = time.perf_counter()
            compiled = _compile_bounded_plan(
                resolved=trainer.resolved_metadata,
                capture=capture,
                parameter_layout=parameter_layout,
                max_staging_bytes=buffer_budget,
                metrics=metrics,
                staging_buffers=staging_buffers,
                total_staging_bytes=max_staging_bytes,
                cache_config=self.cache_config,
            )
            metrics["transfer_planning_s"] = time.perf_counter() - started
            metrics["plan_cache_enabled"] = int(self.cache_config.cache_plan)
            started = time.perf_counter()
            resolved = trainer.resolved_metadata
            required_metadata = {
                agent: metadata
                for agent, metadata in resolved.agent_metadata.items()
                if agent in compiled.required_agents
            }
            self._reset_incompatible_arenas(
                compiled.batches, buffer_budget, staging_device, staging_buffers
            )
            transport = self._connect_sources(
                resolved, required_metadata, host_staging=staging_device == "cpu"
            )
            self._prepare_arenas(
                compiled.batches, staging_device, staging_buffers
            )
            metrics["connection_registration_s"] = time.perf_counter() - started
            descriptors = self._prepare_bounded_descriptors(
                compiled,
                previous_descriptors,
                enabled=bool(metrics["plan_cache_enabled"]),
            )
            prepared = _PreparedBoundedTransfer(
                compiled.batches, resolved.sources, transport, metrics
            )
            cached = _CachedPullPlan(
                trainer,
                capture,
                MappingProxyType(parameter_layout),
                compiled,
                tuple(manifests),
                compiled.required_agents,
            )
            self._publish_prepared(cached, prepared, reusable=False)
            self._descriptor_cache = descriptors
            return prepared

    def _validate_streaming_settings(
        self, max_staging_bytes: int, staging_device: str, staging_buffers: int
    ) -> int:
        if self._device.type != "cuda":
            raise ValueError("bounded NIXL staging requires a CUDA device")
        if staging_device not in ("cuda", "cpu"):
            raise ValueError("staging_device must be 'cuda' or 'cpu'")
        staging_buffers = _require_positive_bytes(staging_buffers, "staging_buffers")
        max_staging_bytes = _require_positive_bytes(
            max_staging_bytes, "max_staging_bytes"
        )
        buffer_budget = max_staging_bytes // staging_buffers
        if buffer_budget <= 0:
            raise ValueError(
                "max_staging_bytes must cover at least one byte per staging buffer"
            )
        return buffer_budget

    def _reset_incompatible_arenas(
        self,
        batches: tuple[_BoundedBatch, ...],
        buffer_budget: int,
        staging_device: str,
        staging_buffers: int,
    ) -> None:
        arena_bytes = max(batch.nbytes for batch in batches)
        if self._staging_arenas and (
            len(self._staging_arenas) != staging_buffers
            or any(
                arena.numel() < arena_bytes or arena.numel() > buffer_budget
                for arena in self._staging_arenas
            )
            or self._staging_device != torch.device(staging_device)
        ):
            mode = self._workspace_mode
            self.reset_workspace()
            assert mode is not None
            self._select_workspace_mode(mode)

    def _prepare_arenas(
        self,
        batches: tuple[_BoundedBatch, ...],
        staging_device: str,
        staging_buffers: int,
    ) -> None:
        arena_bytes = max(batch.nbytes for batch in batches)
        if not self._staging_arenas:
            self._staging_device = torch.device(staging_device)
            try:
                for index in range(staging_buffers):
                    arena = self._allocate_arena(arena_bytes)
                    # Retain storage before registration so failed setup can be cleaned up.
                    self._staging_arenas.append(arena)
                    if staging_device == "cpu":
                        self._staging_registrations.append(
                            self._manager.register_dram_buffer(arena)
                        )
                    else:
                        self._manager.register_tensors(
                            {f"__bounded_arena_{index}__": arena}
                        )
            except Exception:
                self.reset_workspace()
                raise

    def _prepare_bounded_descriptors(
        self,
        compiled: _CompiledBoundedPlan,
        previous: _BoundedDescriptors | None,
        *,
        enabled: bool,
    ) -> _BoundedDescriptors | None:
        if not enabled or not all(
            type(arena) is torch.Tensor for arena in self._staging_arenas
        ):
            return None
        if (
            previous is not None
            and previous.plan is compiled
            and previous.matches(
                compiled.batches, self._workspace_generation, self._staging_arenas
            )
        ):
            return previous
        return _BoundedDescriptors(
            compiled,
            self._workspace_generation,
            tuple(
                (weakref.ref(arena), _arena_geometry(arena))
                for arena in self._staging_arenas
            ),
            (None,) * len(compiled.batches),
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
        metrics["staging_peak_bytes"] = sum(a.numel() for a in arenas)
        metrics["staging_buffers"] = len(arenas)
        metrics["batches"] = len(prepared.batches)
        batches = prepared.batches
        metrics.update(
            descriptor_cache_hits=0, descriptor_cache_misses=0, descriptor_builds=0
        )

        def descriptors(
            index: int,
            recv: dict[str, torch.Tensor],
            full: dict[str, torch.Tensor],
            convert: dict[str, torch.Tensor],
        ) -> tuple[ReadDescriptor | _BoundedReadDescriptor, ...]:
            entry = self._descriptor_cache
            if entry is not None and not entry.matches(
                batches, self._workspace_generation, arenas
            ):
                self._descriptor_cache = entry = None
            if entry is not None and entry.batches[index] is not None:
                metrics["descriptor_cache_hits"] += 1
                return entry.batches[index]
            metrics["descriptor_cache_misses"] += 1
            metrics["descriptor_builds"] += 1
            fresh = tuple(self._descriptors(batches[index].plan, recv, full, convert))
            if entry is None:
                return fresh
            if not all(
                type(item.session) is str
                and type(item.src_addr) is int
                and type(item.dst_addr) is int
                and type(item.nbytes) is int
                for item in fresh
            ):
                self._descriptor_cache = None
                return fresh
            immutable = tuple(
                _BoundedReadDescriptor(
                    item.session, item.src_addr, item.dst_addr, item.nbytes
                )
                for item in fresh
            )
            self._descriptor_cache = replace(
                entry,
                batches=entry.batches[:index]
                + (immutable,)
                + entry.batches[index + 1 :],
            )
            return immutable

        def carve(
            batch: _BoundedBatch, arena: torch.Tensor
        ) -> tuple[dict[str, torch.Tensor], ...]:
            offset = 0
            buffers = []
            for layout in batch.layouts:
                tensors = {}
                for name, (shape, dtype) in layout.items():
                    nbytes = math.prod(shape) * dtype.itemsize
                    tensors[name] = (
                        arena[offset : offset + nbytes].view(dtype).view(shape)
                    )
                    offset += ((nbytes + 255) // 256) * 256
                buffers.append(tensors)
            return tuple(buffers)

        def post(index: int) -> tuple:
            batch = batches[index]
            recv, convert, full = carve(batch, arenas[index % len(arenas)])
            # Captured loaders may leave padding untouched. Reused arenas must
            # reproduce the zero-filled load layout before the NIC writes it.
            for tensors in (recv, convert, full):
                for tensor in tensors.values():
                    tensor.zero_()
            if arenas[index % len(arenas)].device.type == "cuda":
                torch.cuda.synchronize(self._device)
            sources = {
                c.src_name: prepared.sources[c.src_name] for c in batch.capture.copies
            }
            chunk = _PreparedNixlTransfer(
                batch.plan,
                batch.capture,
                sources,
                descriptors(index, recv, full, convert),
                prepared.transport,
            )
            started = time.perf_counter()
            posted = prepared.transport.post_reads(list(chunk.descriptors))
            return chunk, (recv, convert, full), posted, started

        pending = None
        unwinding = False
        completed = False
        try:
            pending = post(0)
            for index in range(len(batches)):
                chunk, buffers, posted, started = pending
                pending = None
                self._recv_buffers, self._convert_buffers, self._full_buffers = buffers
                self._active = chunk
                staged = self._complete_stage(chunk, posted, started)
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
                self._descriptor_cache = None
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
    ) -> None:
        if current:
            return
        with classic_cuda_alloc():
            current.update(
                {
                    name: torch.empty(shape, dtype=dtype, device=self._device)
                    for name, (shape, dtype) in expected.items()
                }
            )

    def _reset_incompatible_full_workspace(self, expected: _StagingLayouts) -> None:
        if any(
            current and self._layout(current) != layout
            for current, layout in zip(
                (self._recv_buffers, self._convert_buffers, self._full_buffers),
                expected,
            )
        ):
            mode = self._workspace_mode
            self.reset_workspace()
            assert mode is not None
            self._select_workspace_mode(mode)

    def _ensure_workspace(self, expected: _StagingLayouts) -> None:
        recv_expected, convert_expected, full_expected = expected
        self._ensure_buffers(self._recv_buffers, recv_expected)
        self._ensure_buffers(self._convert_buffers, convert_expected)
        self._ensure_buffers(self._full_buffers, full_expected)

        recv_params = set(recv_expected)
        if self._registered_recv_params and self._registered_recv_params != recv_params:
            raise RuntimeError(
                "receive parameter set changed; restart the generator engine"
            )
        try:
            if convert_expected and not self._convert_registered:
                self._manager.register_tensors(
                    {
                        f"__convert__{name}": tensor
                        for name, tensor in self._convert_buffers.items()
                    }
                )
                self._convert_registered = True
            if full_expected and not self._full_registered:
                self._manager.register_tensors(
                    {
                        f"__full__{name}": tensor
                        for name, tensor in self._full_buffers.items()
                    }
                )
                self._full_registered = True
            if not self._registered_recv_params and recv_params:
                self._manager.register_tensors(self._recv_buffers)
                self._registered_recv_params = recv_params
        except Exception:
            self.reset_workspace()
            raise

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
        self, prepared: _PreparedNixlTransfer, posted: list, started: float
    ) -> _StagedNixlWeights:
        """Wait for posted READs, then reconstruct, convert, and verify."""
        wait_started = time.perf_counter()
        prepared.transport.await_reads(posted)
        wire_wait_seconds = time.perf_counter() - wait_started
        wire_seconds = time.perf_counter() - started

        reconstruct_started = time.perf_counter()
        for full in prepared.plan.full_pulls:
            source = self._full_buffers[full.src_name]
            for copy in full.copies:
                destination = self._recv_buffers[copy.param_name].as_strided(
                    copy.dest_shape,
                    copy.dest_stride,
                    self._recv_buffers[copy.param_name].storage_offset()
                    + copy.dest_offset,
                )
                destination.copy_(_replay_ops(source, copy.op_chain))
        converted = {convert.param_name for convert in prepared.plan.converts}
        conversion_copies = {
            copy.param_name: copy
            for copy in prepared.capture.copies
            if copy.param_name in converted
        }
        for convert in prepared.plan.converts:
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
        if self.cache_config.publish_digest:
            self._verify(prepared)

        bytes_received = sum(d.nbytes for d in prepared.descriptors)
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
            len(prepared.plan.segments),
            len(prepared.plan.full_pulls),
            len(prepared.plan.converts),
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
                "full_pull_sources": len(prepared.plan.full_pulls),
                "converts": len(prepared.plan.converts),
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
                if not shard.digest:
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
                if actual != shard.digest:
                    raise RuntimeError(
                        f"staged weight digest mismatch for source {name!r} "
                        f"at offset {tuple(shard.shard_offset)}"
                    )

    def close(self) -> None:
        if self._closed:
            return
        self._invalidate_descriptors()
        self._cached_pull_plan = None
        self._generator_layout = None
        self._trainer_snapshot = None
        self._full_copy_descriptors = None
        self._transport = None
        self._closed = True
        self._active = None
        self._loaded_agent_metadata.clear()
        if self._owns_manager:
            self._release_staging_registrations()
            self._manager.shutdown()
            # The agent's registrations are gone, so staging storage can be
            # freed eagerly. A shared agent may still hold these buffers
            # registered; keep them referenced until its owner shuts down.
            self._recv_buffers.clear()
            self._convert_buffers.clear()
            self._full_buffers.clear()
            self._staging_arenas.clear()
            self._staging_device = None


__all__: list[str] = []

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer-memory source resolution."""

import hashlib
import json
import logging
import math
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from time import perf_counter, time_ns

import grpc
from modelexpress import envs, telemetry
from modelexpress.refit.reshard.rendezvous import unwrap_rendezvous_blob
from modelexpress.refit.timing import add_refit_duration, refit_span

from ... import refit_pb2, refit_pb2_grpc
from ...control import WeightVersion
from ...train.manifest import ShardChecksumKey, version_metadata_digest
from ...train import WeightPayloadFormat
from ..adapter import TrainerSourceShard
from ..plan import (
    ResolvedSource,
    ResolvedTrainerSource,
    SourceResolver,
    TrainerSourceSnapshot,
    WeightSource,
)

logger = logging.getLogger("modelexpress_rl.inference.source.trainer")

_MAX_MANIFEST_MESSAGE_SIZE_BYTES = 100 * 1024 * 1024


class _SlotReplicas:
    """Replicas published for one source slot, resolved as far as asked.

    Health is decided per slot and independently of every other slot. Pairing
    replicas by a shared offset instead means a fleet whose healthy replicas
    sit at different offsets in different slots yields no candidate at all,
    even though a complete healthy set exists: with only ``a0`` up in one slot
    and only ``b1`` in the next, the aligned pairs ``(a0, b0)`` and
    ``(a1, b1)`` both contain a dead source and ``(a0, b1)`` is never tried.

    Resolution stays lazy because it is charged per manifest fetched, and the
    first replica of each slot is all a healthy fleet ever needs.
    """

    def __init__(
        self,
        slot_id: str,
        shards: list[refit_pb2.WeightVersionShard],
        resolve: Callable[[refit_pb2.WeightVersionShard], TrainerSourceShard],
    ) -> None:
        self.slot_id = slot_id
        self._pending = list(shards)
        self._resolve = resolve
        self._usable: list[TrainerSourceShard] = []

    def usable(self, index: int) -> TrainerSourceShard | None:
        """Return the index-th usable replica, resolving no further than needed."""
        while len(self._usable) <= index and self._pending:
            shard = self._pending.pop(0)
            try:
                self._usable.append(self._resolve(shard))
            except (grpc.RpcError, RuntimeError) as error:
                logger.warning(
                    "trainer source %s failed for slot %s: %s",
                    shard.worker_id,
                    self.slot_id,
                    error,
                )
        if index < len(self._usable):
            return self._usable[index]
        return None

    @property
    def exhausted(self) -> bool:
        """Whether every published replica has been tried."""
        return not self._pending

    @property
    def usable_count(self) -> int:
        """How many replicas resolved so far; final once exhausted."""
        return len(self._usable)


class TrainerSourceResolver(SourceResolver):
    """Resolve trainer metadata and round checksums without compiling a plan."""

    def __init__(
        self,
        *,
        service: Callable[[], refit_pb2_grpc.RefitServiceStub],
        rpc_timeout_seconds: float,
        cached_source: Callable[[], TrainerSourceSnapshot | None] | None = None,
    ) -> None:
        self._service = service
        self._cached_source = cached_source
        self._rpc_timeout_seconds = rpc_timeout_seconds


    @property
    def kind(self) -> WeightSource:
        return WeightSource.TRAINER

    def supports(self, version: WeightVersion) -> bool:
        return version.payload_format is WeightPayloadFormat.FULL_TENSOR

    def payload_format(self, version: WeightVersion) -> WeightPayloadFormat:
        return version.payload_format

    def candidates(self, version: WeightVersion) -> Iterator[ResolvedSource]:
        if version.trainer_mesh_id is None:
            raise RuntimeError("trainer publication requires trainer_mesh_id")
        try:
            mesh_response = self._service().GetTrainerMesh(
                refit_pb2.GetTrainerMeshRequest(mesh_id=version.trainer_mesh_id),
                timeout=self._rpc_timeout_seconds,
            )
        except grpc.RpcError as error:
            logger.warning(
                "trainer mesh lookup failed for version %s: %s",
                version.version_id,
                error,
            )
            return
        if not mesh_response.HasField("mesh"):
            raise RuntimeError("MX GetTrainerMesh response is missing mesh")
        mesh = mesh_response.mesh
        if mesh.mesh_id != version.trainer_mesh_id:
            raise RuntimeError("MX GetTrainerMesh returned a different mesh ID")
        if mesh.generation != version.trainer_mesh_generation:
            raise RuntimeError(
                "weight version trainer mesh generation differs from the current mesh"
            )
        expected_slots = tuple(sorted({
            metadata.logical_shard_id for metadata in mesh_response.mesh.workers.values()
        }))
        mesh_workers = mesh_response.mesh.workers
        mesh_generation = version.trainer_mesh_generation
        try:
            with refit_span(
                "source_preparation",
                metadata={"manifest_list_count": 1},
                accumulate_metadata=True,
                duration_key="manifest_list_s",
            ) as counters:
                response = self._service().ListWeightVersionShards(
                    refit_pb2.ListWeightVersionShardsRequest(
                        version_id=version.version_id
                    ),
                    timeout=self._rpc_timeout_seconds,
                )
                counters["published_shard_count"] = len(response.shards)
        except grpc.RpcError as error:
            logger.warning(
                "trainer source discovery failed for version %s: %s",
                version.version_id,
                error,
            )
            return
        published = defaultdict(list)
        for shard in response.shards:
            if shard.version_id != version.version_id:
                raise RuntimeError("publication listing returned a different weight version")
            metadata = mesh_workers.get(shard.worker_id)
            if metadata is None or metadata.logical_shard_id != shard.logical_shard_id:
                continue
            published[shard.logical_shard_id].append(shard)

        cached = self._cached_source() if self._cached_source is not None else None
        stable_sources = {
            (source.metadata_endpoint, source.stable_metadata_digest): source
            for source in cached.shards
        } if cached is not None else {}
        round_sources = {}
        failed_sources = set()
        round_checksums = {}
        verify = envs.MX_RESHARD_PUBLISH_DIGEST
        counters: dict[str, int | float] = defaultdict(int)
        resolution_seconds = 0.0
        resolution_failed = False
        resolution_trace_started = None

        def flush() -> None:
            nonlocal resolution_seconds, resolution_failed, resolution_trace_started
            if counters:
                add_refit_duration(
                    "source_preparation",
                    resolution_seconds,
                    status="error" if resolution_failed else "ok",
                    metadata=dict(counters),
                    accumulate_metadata=True,
                )
                if resolution_trace_started is not None:
                    telemetry.completed_span(
                        "mx.refit.source_resolution",
                        resolution_trace_started,
                        time_ns(),
                        {**counters, "status": "error" if resolution_failed else "ok"},
                    )
                    resolution_trace_started = None
                counters.clear()
                resolution_seconds = 0.0
                resolution_failed = False

        def resolve(shard: refit_pb2.WeightVersionShard) -> TrainerSourceShard:
            identity = (shard.logical_shard_id, shard.worker_id)
            if identity in round_sources:
                return round_sources[identity]
            key = (shard.metadata_endpoint, shard.stable_metadata_digest)
            nonlocal resolution_seconds, resolution_failed, resolution_trace_started
            if (
                resolution_trace_started is None
                and (key not in stable_sources or verify)
                and telemetry.recording()
            ):
                resolution_trace_started = time_ns()
            started = perf_counter()
            try:
                source, checksums = self._resolve_source(
                    shard, stable_sources.get(key), verify=verify, counters=counters
                )
            except BaseException:
                resolution_failed = True
                raise
            finally:
                resolution_seconds += perf_counter() - started
            stable_sources[key] = source
            round_sources[identity] = source
            round_checksums[identity] = checksums
            return source

        try:
            selection = None
            if (
                cached is not None
                and cached.mesh_id == version.trainer_mesh_id
                and cached.mesh_generation == mesh_generation
                and tuple(sorted(source.source_slot_id for source in cached.shards)) == expected_slots
            ):
                matching = []
                for source in cached.shards:
                    member = mesh_workers.get(source.worker_id)
                    publication = next((
                        item for item in published[source.source_slot_id]
                        if item.worker_id == source.worker_id
                        and item.metadata_endpoint == source.metadata_endpoint
                        and item.stable_metadata_digest == source.stable_metadata_digest
                    ), None)
                    if (
                        member is None
                        or member.logical_shard_id != source.source_slot_id
                        or member.metadata_endpoint != source.metadata_endpoint
                        or publication is None
                    ):
                        break
                    matching.append(publication)
                if len(matching) == len(cached.shards):
                    try:
                        for publication in matching:
                            resolve(publication)
                    except (grpc.RpcError, RuntimeError) as error:
                        failed_sources.add((publication.logical_shard_id, publication.worker_id))
                        logger.warning("cached trainer source %s failed: %s", publication.worker_id, error)
                    else:
                        if verify:
                            self._check_current_mesh(version.trainer_mesh_id, mesh_generation)
                        selection = tuple(
                            (source.source_slot_id, source.worker_id) for source in cached.shards
                        )
                        flush()
                        yield ResolvedTrainerSource(
                            snapshot=cached,
                            checksums={
                                key: digest for identity in selection
                                for key, digest in round_checksums[identity].items()
                            },
                        )

            slots = []
            for source_slot_id in expected_slots:
                ordered = sorted(
                    (item for item in published[source_slot_id]
                     if (item.logical_shard_id, item.worker_id) not in failed_sources),
                    key=lambda item: item.worker_id
                )
                if not ordered:
                    logger.warning(
                        "no trainer source published for required slot %s",
                        source_slot_id,
                    )
                    return
                slots.append(_SlotReplicas(source_slot_id, ordered, resolve))

            seen: set[tuple[tuple[str, str], ...]] = (
                {selection} if selection is not None else set()
            )
            offset = 0
            while True:
                selected = []
                for slot in slots:
                    source = slot.usable(offset)
                    if source is None:
                        if not slot.usable_count:
                            logger.warning(
                                "no usable trainer source for required slot %s",
                                slot.slot_id,
                            )
                            return
                        # Exhausted and shorter than the candidate index, so cycle
                        # its healthy replicas rather than give up on a slot that
                        # simply has fewer of them.
                        source = slot.usable(offset % slot.usable_count)
                    selected.append(source)
                selection = tuple(
                    (source.source_slot_id, source.worker_id) for source in selected
                )
                if selection not in seen:
                    self._check_current_mesh(version.trainer_mesh_id, mesh_generation)
                    seen.add(selection)
                    flush()
                    yield ResolvedTrainerSource(
                        snapshot=TrainerSourceSnapshot(
                            mesh_id=version.trainer_mesh_id,
                            mesh_generation=mesh_generation,
                            shards=tuple(selected),
                        ),
                        checksums={
                            key: digest
                            for source in selected
                            for key, digest in round_checksums[
                                (source.source_slot_id, source.worker_id)
                            ].items()
                        },
                    )
                deepest = max((slot.usable_count for slot in slots), default=1)
                if all(slot.exhausted for slot in slots) and offset + 1 >= deepest:
                    return
                offset += 1
        finally:
            flush()

    def _check_current_mesh(self, mesh_id: str, generation: int) -> None:
        current = self._service().GetTrainerMesh(
            refit_pb2.GetTrainerMeshRequest(mesh_id=mesh_id),
            timeout=self._rpc_timeout_seconds,
        )
        if (
            not current.HasField("mesh")
            or current.mesh.mesh_id != mesh_id
            or current.mesh.generation != generation
        ):
            raise RuntimeError("trainer mesh generation changed during source resolution")

    def _resolve_source(
        self,
        shard: refit_pb2.WeightVersionShard,
        cached: TrainerSourceShard | None,
        *,
        verify: bool,
        counters: dict[str, int | float],
    ) -> tuple[TrainerSourceShard, dict[ShardChecksumKey, str]]:
        if not shard.metadata_endpoint:
            raise RuntimeError("NIXL source is missing its metadata endpoint")
        if not shard.stable_metadata_digest:
            raise RuntimeError("source is missing its stable metadata digest")
        checksums = {}
        counters["manifest_cache_hits"] += int(cached is not None)
        counters["manifest_cache_misses"] += int(cached is None)
        if cached is None or verify:
            with grpc.insecure_channel(
                shard.metadata_endpoint,
                options=[
                    ("grpc.max_receive_message_length", _MAX_MANIFEST_MESSAGE_SIZE_BYTES)
                ],
            ) as channel:
                stub = refit_pb2_grpc.RefitWorkerServiceStub(channel)
                if cached is None:
                    blob, keys = self._fetch_stable_metadata(stub, shard, counters)
                    cached = TrainerSourceShard(
                        source_slot_id=shard.logical_shard_id,
                        worker_id=shard.worker_id,
                        stable_metadata_digest=shard.stable_metadata_digest,
                        metadata_endpoint=shard.metadata_endpoint,
                        metadata=blob,
                        checksum_keys=keys,
                    )
                    counters["manifest_fetch_bytes"] += len(blob)
                    counters["manifest_fetch_count"] += 1
                if verify:
                    checksums = self._fetch_version_metadata(stub, shard, cached.checksum_keys)
                    counters["version_metadata_fetch_count"] += 1
        counters["manifest_bytes"] += len(cached.metadata)
        if cached.source_slot_id != shard.logical_shard_id or cached.worker_id != shard.worker_id:
            cached = TrainerSourceShard(
                source_slot_id=shard.logical_shard_id,
                worker_id=shard.worker_id,
                stable_metadata_digest=shard.stable_metadata_digest,
                metadata_endpoint=shard.metadata_endpoint,
                metadata=cached.metadata,
                checksum_keys=cached.checksum_keys,
            )
        return cached, checksums

    def _fetch_stable_metadata(
        self,
        stub: refit_pb2_grpc.RefitWorkerServiceStub,
        shard: refit_pb2.WeightVersionShard,
        counters: dict[str, int | float],
    ) -> tuple[bytes, dict[tuple[str, int], ShardChecksumKey]]:
        started = perf_counter()
        try:
            response = stub.GetTrainerShardMetadata(
                refit_pb2.GetTrainerShardMetadataRequest(
                    metadata_digest=shard.stable_metadata_digest
                ),
                timeout=self._rpc_timeout_seconds,
            )
        finally:
            counters["manifest_fetch_s"] += perf_counter() - started
        blob = response.metadata
        if (
            response.metadata_digest != shard.stable_metadata_digest
            or hashlib.sha256(blob).hexdigest() != shard.stable_metadata_digest
        ):
            raise RuntimeError(f"stable metadata digest mismatch for source slot {shard.logical_shard_id!r}")
        try:
            payload = json.loads(blob)
            if (
                "publisher_step" in payload
                or type(payload["tensor_count"]) is not int
                or payload["tensor_count"] <= 0
                or type(payload["total_bytes"]) is not int
                or payload["total_bytes"] <= 0
                or any("digest" in item for tensor in payload["tensors"] for item in tensor["shards"])
            ):
                raise ValueError("stable metadata contains version state or invalid counts")
            tensors = unwrap_rendezvous_blob(blob).tensors
            keys = {
                (tensor.name, index): (
                    tensor.name, item.agent_name, item.addr,
                    tuple(item.shard_offset), tuple(item.shape),
                )
                for tensor in tensors for index, item in enumerate(tensor.shards)
            }
            if len({tensor.name for tensor in tensors}) != len(tensors):
                raise ValueError("stable metadata contains duplicate tensor names")
            if (
                payload["tensor_count"] != len(tensors)
                or payload["total_bytes"] != sum(
                    math.prod(item.shape) * tensor.elsize
                    for tensor in tensors for item in tensor.shards
                )
                or len(set(keys.values())) != len(keys)
            ):
                raise ValueError("stable metadata counts or physical shard identities are inconsistent")
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"invalid stable metadata for source slot {shard.logical_shard_id!r}") from error
        return blob, keys

    def _fetch_version_metadata(
        self,
        stub: refit_pb2_grpc.RefitWorkerServiceStub,
        shard: refit_pb2.WeightVersionShard,
        keys: Mapping[tuple[str, int], ShardChecksumKey],
    ) -> dict[ShardChecksumKey, str]:
        if not shard.version_metadata_digest:
            raise RuntimeError("source did not publish version verification metadata")
        response = stub.GetWeightVersionShardMetadata(
            refit_pb2.GetWeightVersionShardMetadataRequest(
                version_id=shard.version_id, logical_shard_id=shard.logical_shard_id
            ),
            timeout=self._rpc_timeout_seconds,
        )
        metadata = response.metadata
        try:
            digest = version_metadata_digest(metadata)
        except ValueError as error:
            raise RuntimeError("invalid version checksum metadata") from error
        if (
            not response.HasField("metadata")
            or response.version_metadata_digest != shard.version_metadata_digest
            or digest != shard.version_metadata_digest
            or metadata.version_id != shard.version_id
            or metadata.worker_id != shard.worker_id
            or metadata.logical_shard_id != shard.logical_shard_id
            or metadata.stable_metadata_digest != shard.stable_metadata_digest
        ):
            raise RuntimeError("version checksum metadata identity or digest mismatch")
        published = {(row.tensor_name, row.shard_index): row.digest for row in metadata.checksums}
        if published.keys() != keys.keys() or any(not digest for digest in published.values()):
            raise RuntimeError("version checksum metadata does not cover the stable source shards")
        return {keys[key]: digest for key, digest in published.items()}


__all__ = ["TrainerSourceResolver"]

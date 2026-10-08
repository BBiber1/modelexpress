# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer-memory source resolution."""

import hashlib
import logging
from collections import defaultdict
from collections.abc import Callable, Iterator

import grpc
from modelexpress import envs
from modelexpress.refit.reshard.rendezvous import structural_manifest_digest
from modelexpress.refit.timing import refit_span

from ... import refit_pb2, refit_pb2_grpc
from ...control import WeightVersion
from ...train import WeightPayloadFormat
from ..adapter import TrainerSourceShard
from ..plan import ResolvedSource, SourceResolver, TrainerSourceSnapshot, WeightSource

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
    """Resolve trainer shard manifests without compiling a transfer plan."""

    def __init__(
        self,
        *,
        service: Callable[[], refit_pb2_grpc.RefitServiceStub],
        rpc_timeout_seconds: float,
        cached_source: Callable[[], TrainerSourceSnapshot | None] | None = None,
    ) -> None:
        self._service = service
        self._rpc_timeout_seconds = rpc_timeout_seconds
        self._cached_source = cached_source

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
        if version.trainer_mesh_generation <= 0:
            raise RuntimeError(
                "trainer publication requires a positive trainer_mesh_generation; recreate the version"
            )
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
        cached = self._cached_source() if self._cached_source is not None else None
        same_mesh = cached is not None and (mesh.mesh_id, mesh.generation) == (
            cached.mesh_id,
            cached.mesh_generation,
        )
        diagnostic = envs.MX_REFIT_DEBUG_VALIDATE_PLAN or envs.MX_RESHARD_PUBLISH_DIGEST
        if (
            same_mesh
            and envs.MX_REFIT_CACHE_RESOLVED_SOURCES
            and envs.MX_REFIT_CACHE_BOUNDED_PLANS
            and not diagnostic
        ):
            yield cached
            # A resumed candidate means preparation failed: discover fresh replicas.
        expected_slots = tuple(
            sorted(
                {
                    metadata.logical_shard_id
                    for metadata in mesh_response.mesh.workers.values()
                }
            )
        )
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
            metadata = mesh_workers.get(shard.worker_id)
            if metadata is None or metadata.logical_shard_id != shard.logical_shard_id:
                continue
            published[shard.logical_shard_id].append(shard)

        counters: dict[str, int | float] = {}
        slots = []
        for source_slot_id in expected_slots:
            ordered = sorted(published[source_slot_id], key=lambda item: item.worker_id)
            if not ordered:
                logger.warning(
                    "no trainer source published for required slot %s",
                    source_slot_id,
                )
                return
            slots.append(
                _SlotReplicas(
                    source_slot_id,
                    ordered,
                    lambda shard: self._resolve_source(shard, counters),
                )
            )

        seen: set[tuple[tuple[str, str], ...]] = set()
        offset = 0
        while True:
            with refit_span(
                "source_preparation",
                metadata={
                    "manifest_fetch_count": 0,
                    "manifest_fetch_bytes": 0,
                    "manifest_bytes": 0,
                },
                accumulate_metadata=True,
                duration_key="source_resolution_s",
            ) as counters:
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
                # Fetching worker manifests can outlive the original mesh snapshot.
                current = self._service().GetTrainerMesh(
                    refit_pb2.GetTrainerMeshRequest(mesh_id=version.trainer_mesh_id),
                    timeout=self._rpc_timeout_seconds,
                )
                if (
                    not current.HasField("mesh")
                    or current.mesh.mesh_id != version.trainer_mesh_id
                    or current.mesh.generation != mesh_generation
                ):
                    raise RuntimeError(
                        "trainer mesh generation changed during source resolution"
                    )
                seen.add(selection)
                yield TrainerSourceSnapshot(
                    mesh_id=version.trainer_mesh_id,
                    mesh_generation=mesh_generation,
                    shards=tuple(selected),
                )
            deepest = max((slot.usable_count for slot in slots), default=1)
            if all(slot.exhausted for slot in slots) and offset + 1 >= deepest:
                return
            offset += 1

    def _resolve_source(
        self, shard: refit_pb2.WeightVersionShard, counters: dict[str, int | float]
    ) -> TrainerSourceShard:
        if not shard.manifest_endpoint:
            raise RuntimeError("NIXL source is missing its manifest endpoint")
        if not shard.manifest_digest:
            raise RuntimeError("source is missing its manifest digest")
        counters["manifest_fetch_count"] = counters.get("manifest_fetch_count", 0) + 1
        manifest, structure_digest = self._fetch_manifest(shard)
        counters["manifest_fetch_bytes"] = counters.get(
            "manifest_fetch_bytes", 0
        ) + len(manifest)
        counters["manifest_bytes"] = counters.get("manifest_bytes", 0) + len(manifest)
        return TrainerSourceShard(
            source_slot_id=shard.logical_shard_id,
            worker_id=shard.worker_id,
            manifest_digest=shard.manifest_digest,
            manifest_endpoint=shard.manifest_endpoint,
            manifest=manifest,
            structural_digest=structure_digest,
        )

    def _fetch_manifest(self, shard: refit_pb2.WeightVersionShard) -> tuple[bytes, str]:
        """Fetch, verify and fingerprint one worker's manifest.

        Three spans on one stage rather than one, because the stage total
        cannot say whether a slow warm refit is waiting on the wire or on the
        CPU, and with digests published these manifests are refetched by
        construction on every version.
        """
        with (
            refit_span(
                "source_preparation",
                accumulate_metadata=True,
                duration_key="manifest_fetch_s",
            ),
            grpc.insecure_channel(
                shard.manifest_endpoint,
                options=[
                    (
                        "grpc.max_receive_message_length",
                        _MAX_MANIFEST_MESSAGE_SIZE_BYTES,
                    )
                ],
            ) as channel,
        ):
            response = refit_pb2_grpc.RefitWorkerServiceStub(
                channel
            ).GetWeightVersionShardManifest(
                refit_pb2.GetWeightVersionShardManifestRequest(
                    version_id=shard.version_id,
                    logical_shard_id=shard.logical_shard_id,
                ),
                timeout=self._rpc_timeout_seconds,
            )
        with refit_span(
            "source_preparation",
            accumulate_metadata=True,
            duration_key="manifest_hash_s",
        ):
            digest = hashlib.sha256(response.manifest).hexdigest()
        if (
            response.manifest_digest != shard.manifest_digest
            or digest != shard.manifest_digest
        ):
            raise RuntimeError(
                f"manifest digest mismatch for source slot {shard.logical_shard_id!r}"
            )
        try:
            with refit_span(
                "source_preparation",
                accumulate_metadata=True,
                duration_key="manifest_fingerprint_s",
            ):
                structure_digest = structural_manifest_digest(response.manifest)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"invalid manifest for source slot {shard.logical_shard_id!r}"
            ) from error
        return response.manifest, structure_digest


__all__ = ["TrainerSourceResolver"]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-local serving for versioned trainer manifests."""

from __future__ import annotations

import hashlib
import json
import threading

import grpc

from .. import refit_pb2, refit_pb2_grpc
from .adapter import TrainerShardMetadata
from ..shard_metadata import version_metadata_digest


def bound_tensor_manifest(tensor_coverage: list[dict]) -> bytes:
    """Canonical address- and content-independent coverage of one binding."""
    tensors = []
    for tensor in tensor_coverage:
        shards = sorted(
            (
                {"shard_offset": shard["shard_offset"], "shape": shard["shape"]}
                for shard in tensor["shards"]
            ),
            key=lambda shard: (shard["shard_offset"], shard["shape"]),
        )
        tensors.append(
            {
                "name": tensor["name"],
                "dtype": tensor["dtype"],
                "elsize": tensor["elsize"],
                "full_shape": tensor["full_shape"],
                "shards": shards,
            }
        )
    tensors.sort(key=lambda tensor: tensor["name"])
    return json.dumps(
        {"tensors": tensors}, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


class RefitWorkerService(refit_pb2_grpc.RefitWorkerServiceServicer):
    """Serve process-lifetime stable metadata and releasable version checksums."""

    def __init__(self, *, endpoint: str) -> None:
        if not endpoint.strip():
            raise ValueError("endpoint is required")
        self.endpoint = endpoint
        self._metadata: dict[str, bytes] = {}
        self._versions: dict[tuple[str, str], bytes] = {}
        self._lock = threading.Lock()

    def publish_binding(self, manifest: bytes) -> None:
        binding_id = hashlib.sha256(manifest).hexdigest()
        with self._lock:
            self._metadata[binding_id] = manifest

    def publish_metadata(
        self,
        *,
        version_id: str,
        logical_shard_id: str,
        metadata: TrainerShardMetadata,
        version_metadata: refit_pb2.WeightVersionShardMetadata | None = None,
    ) -> str:
        """Publish stable metadata and optional version checksums before returning."""
        if not version_id.strip():
            raise ValueError("version_id is required")
        if not logical_shard_id.strip():
            raise ValueError("logical_shard_id is required")
        encoded = None
        if version_metadata is not None:
            if (
                version_metadata.version_id != version_id
                or version_metadata.logical_shard_id != logical_shard_id
                or version_metadata.stable_metadata_digest != metadata.digest
                or not version_metadata.worker_id
            ):
                raise ValueError("version metadata does not identify the published shard")
            version_metadata_digest(version_metadata)
            encoded = version_metadata.SerializeToString(deterministic=True)
        key = (version_id, logical_shard_id)
        with self._lock:
            if encoded is not None:
                existing = self._versions.get(key)
                if existing is not None and existing != encoded:
                    raise ValueError(
                        "different version metadata is already published for "
                        f"version_id={version_id!r}, logical_shard_id={logical_shard_id!r}"
                    )
                self._versions[key] = encoded
            self._metadata[metadata.digest] = metadata.data
        return self.endpoint

    def release_version_metadata(self, *, version_id: str, logical_shard_id: str) -> None:
        """Drop version checksums after the protected version is released."""
        key = (version_id, logical_shard_id)
        with self._lock:
            self._versions.pop(key, None)

    def GetTrainerShardMetadata(
        self, request, context
    ) -> refit_pb2.GetTrainerShardMetadataResponse:
        with self._lock:
            metadata = self._metadata.get(request.metadata_digest)
        if metadata is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "metadata was not found")
        return refit_pb2.GetTrainerShardMetadataResponse(
            metadata=metadata, metadata_digest=request.metadata_digest
        )

    def GetWeightVersionShardMetadata(
        self, request, context
    ) -> refit_pb2.GetWeightVersionShardMetadataResponse:
        with self._lock:
            encoded = self._versions.get((request.version_id, request.logical_shard_id))
        if encoded is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "version metadata was not found")
        metadata = refit_pb2.WeightVersionShardMetadata.FromString(encoded)
        return refit_pb2.GetWeightVersionShardMetadataResponse(
            metadata=metadata, version_metadata_digest=version_metadata_digest(metadata)
        )


__all__ = ["RefitWorkerService"]

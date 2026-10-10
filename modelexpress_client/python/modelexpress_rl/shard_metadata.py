# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable source metadata and canonical version checksum records."""

import hashlib
import json

from . import refit_pb2

ShardChecksumKey = tuple[str, str, int, tuple[int, ...], tuple[int, ...]]


def stable_metadata_blob(blob: bytes, tensor_count: int, total_bytes: int) -> bytes:
    payload = json.loads(blob)
    payload.pop("publisher_step", None)
    for tensor in payload["tensors"]:
        for shard in tensor["shards"]:
            shard.pop("digest", None)
    payload["tensor_count"] = tensor_count
    payload["total_bytes"] = total_bytes
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def version_metadata(
    *,
    version_id: str,
    worker_id: str,
    logical_shard_id: str,
    stable_metadata_digest: str,
    checksums: tuple[tuple[str, int, str], ...],
) -> refit_pb2.WeightVersionShardMetadata:
    return refit_pb2.WeightVersionShardMetadata(
        version_id=version_id,
        worker_id=worker_id,
        logical_shard_id=logical_shard_id,
        stable_metadata_digest=stable_metadata_digest,
        checksums=[
            refit_pb2.ShardChecksum(tensor_name=name, shard_index=index, digest=digest)
            for name, index, digest in sorted(checksums)
        ],
    )


def version_metadata_digest(metadata: refit_pb2.WeightVersionShardMetadata) -> str:
    keys = [(row.tensor_name, row.shard_index) for row in metadata.checksums]
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise ValueError("version checksums must have unique canonical tensor/shard keys")
    return hashlib.sha256(metadata.SerializeToString(deterministic=True)).hexdigest()

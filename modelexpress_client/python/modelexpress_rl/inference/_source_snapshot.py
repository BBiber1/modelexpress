# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable host metadata owned by the manifest-byte cache."""

from types import MappingProxyType
from typing import NamedTuple

import torch


class _ShardSnapshot(NamedTuple):
    shard_offset: tuple
    shape: tuple
    session: str
    addr: int
    elsize: int
    digest: str | None


class _TensorSnapshot(NamedTuple):
    global_shape: tuple
    dtype: torch.dtype
    elsize: int
    shards: tuple


def _source_structure(source) -> tuple:
    """Fields baked into a physical plan; per-version content digests are separate."""
    return (
        source.dtype,
        tuple(source.global_shape),
        source.elsize,
        tuple(
            (
                shard.session,
                shard.addr,
                shard.elsize,
                tuple(shard.shard_offset),
                tuple(shard.shape),
            )
            for shard in source.shards
        ),
    )


def _freeze_sources(source_map: dict) -> MappingProxyType:
    return MappingProxyType(
        {
            name: _TensorSnapshot(
                tuple(source.global_shape),
                source.dtype,
                source.elsize,
                tuple(
                    _ShardSnapshot(
                        tuple(shard.shard_offset),
                        tuple(shard.shape),
                        shard.session,
                        shard.addr,
                        shard.elsize,
                        shard.digest,
                    )
                    for shard in source.shards
                ),
            )
            for name, source in source_map.items()
        }
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generator-engine boundary for ModelExpress RL refit installation."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass


class GeneratorEngineContext(ABC):
    """Typed rank-local inputs used to construct one engine adapter."""


@dataclass(frozen=True)
class TrainerSourceShard:
    """Immutable manifest and worker selection for one trainer slot."""

    source_slot_id: str
    worker_id: str
    stable_metadata_digest: str
    metadata_endpoint: str
    metadata: bytes

    @property
    def physical_fingerprint(self) -> tuple:
        return ("NIXL", self.metadata_endpoint, self.stable_metadata_digest)


__all__ = ["GeneratorEngineContext", "TrainerSourceShard"]

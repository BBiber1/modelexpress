# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inputs used by legacy fake engine adapters in receiver tests."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratorTransferInputs:
    version_id: str
    base_version_id: str | None
    layout_signature: str
    payload_format: object
    sources: tuple
    object_storage: object = None
    trainer_mesh_id: str | None = None
    trainer_mesh_generation: int | None = None

    @classmethod
    def from_trainer(cls, version, source) -> "GeneratorTransferInputs":
        return cls(
            version.version_id,
            version.base_version_id,
            version.layout_signature,
            version.payload_format,
            source.shards,
            trainer_mesh_id=source.mesh_id,
            trainer_mesh_generation=source.mesh_generation,
        )

    @property
    def physical_fingerprint(self) -> tuple:
        return (
            self.base_version_id,
            self.layout_signature,
            self.payload_format,
            self.object_storage,
            self.trainer_mesh_id,
            self.trainer_mesh_generation,
            tuple(
                (source.source_slot_id, source.worker_id, source.physical_fingerprint)
                for source in self.sources
            ),
        )

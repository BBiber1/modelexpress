# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from modelexpress import envs


@dataclass(frozen=True)
class RefitCacheConfig:
    cache_plan: bool = field(default_factory=lambda: envs.MX_REFIT_CACHE_PLAN)
    validate_plan: bool = field(
        default_factory=lambda: envs.MX_REFIT_DEBUG_VALIDATE_PLAN
    )
    publish_digest: bool = field(default_factory=lambda: envs.MX_RESHARD_PUBLISH_DIGEST)
    pack_modules: bool = field(default_factory=lambda: envs.MX_REFIT_PACK_MODULES)

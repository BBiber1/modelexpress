# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from modelexpress_rl.inference.nixl_staged_transfer import _PreparedBoundedTransfer
from modelexpress_rl.inference.plan import TrainerSourceSnapshot

from tests.test_refit_bounded_descriptors import _check_values, harness


def prepare_warm(state) -> _PreparedBoundedTransfer:
    return state.transfer.prepare_streaming(
        manifests=None,
        trainer_snapshot=state.transfer.cached_trainer_source(),
        capture_layout=lambda _: (state.capture, state.layout),
        max_staging_bytes=1024,
        staging_device="cpu",
    )


def test_default_warm_refits_reuse_layout_and_read_new_weight_values(
    harness, monkeypatch
) -> None:
    monkeypatch.delenv("MX_REFIT_CACHE_RESOLVED_SOURCES")
    monkeypatch.delenv("MX_REFIT_CACHE_BOUNDED_PLANS")
    first = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    _check_values(harness, harness.collect(first)[1])
    # A normal refit uses the captured generator layout until the mesh changes.
    harness.capture.copies.clear()
    harness.layout.clear()
    for _ in range(3):
        for tensor in harness.sources.values():
            tensor.add_(1)
        prepared = prepare_warm(harness)
        assert prepared.metrics["plan_cache_hits"] == 1
        _check_values(harness, harness.collect(prepared)[1])


@pytest.mark.parametrize("mesh_id,generation", [("other", 1), ("mesh", 2)])
def test_mesh_identity_changes_rebuild_then_reuse(harness, mesh_id, generation) -> None:
    first = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    harness.collect(first)
    for tensor in harness.sources.values():
        tensor.add_(2)
    rebuilt = harness.prepare(
        trainer_snapshot=TrainerSourceSnapshot(mesh_id, generation, ())
    )
    assert rebuilt.metrics["plan_cache_hits"] == 0
    _check_values(harness, harness.collect(rebuilt)[1])
    reused = prepare_warm(harness)
    assert reused.metrics["plan_cache_hits"] == 1
    _check_values(harness, harness.collect(reused)[1])


@pytest.mark.parametrize("plan_debug", [False, True])
@pytest.mark.parametrize("layout_debug", [False, True])
def test_debug_validation_accepts_changed_values_with_stable_structure(
    harness, monkeypatch, plan_debug, layout_debug
) -> None:
    first = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    harness.collect(first)
    monkeypatch.setenv("MX_REFIT_DEBUG_VALIDATE_PLAN", str(int(plan_debug)))
    monkeypatch.setenv(
        "MX_REFIT_DEBUG_VALIDATE_GENERATOR_LAYOUT", str(int(layout_debug))
    )
    for tensor in harness.sources.values():
        tensor.add_(3)
    prepared = harness.prepare(
        trainer_snapshot=harness.transfer.cached_trainer_source()
    )
    assert prepared.metrics["plan_cache_hits"] == 1
    _check_values(harness, harness.collect(prepared)[1])


def test_debug_layout_drift_fails_before_transfer(harness, monkeypatch) -> None:
    first = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    harness.collect(first)
    monkeypatch.setenv("MX_REFIT_DEBUG_VALIDATE_GENERATOR_LAYOUT", "1")
    harness.capture.copies[0].dest_offset += 1
    with pytest.raises(ValueError, match="generator layout changed"):
        harness.prepare(trainer_snapshot=harness.transfer.cached_trainer_source())


def test_debug_source_drift_fails_and_next_attempt_rebuilds(harness, monkeypatch) -> None:
    first = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    harness.collect(first)
    monkeypatch.setenv("MX_REFIT_DEBUG_VALIDATE_PLAN", "1")
    harness.sources["exact"] = harness.sources["exact"].clone().add_(5)
    with pytest.raises(ValueError, match="source metadata changed"):
        harness.prepare(trainer_snapshot=harness.transfer.cached_trainer_source())
    rebuilt = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    assert rebuilt.metrics["plan_cache_hits"] == 0
    _check_values(harness, harness.collect(rebuilt)[1])


@pytest.mark.parametrize(
    "flag", ["MX_REFIT_CACHE_RESOLVED_SOURCES", "MX_REFIT_CACHE_BOUNDED_PLANS"]
)
def test_explicit_cache_disable_requires_validated_preparation(
    harness, monkeypatch, flag
) -> None:
    first = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    harness.collect(first)
    monkeypatch.setenv(flag, "0")
    prepared = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    assert prepared.metrics["plan_cache_hits"] == 0
    _check_values(harness, harness.collect(prepared)[1])


def test_replacing_candidate_under_same_mesh_rebuilds(harness) -> None:
    first = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    harness.collect(first)
    replacement = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    assert replacement.metrics["plan_cache_hits"] == 0
    _check_values(harness, harness.collect(replacement)[1])


def test_mesh_change_during_setup_rejects_then_rebuilds(harness, monkeypatch) -> None:
    current_generation = 2

    def verify_mesh(source) -> None:
        if source.mesh_generation != current_generation:
            raise RuntimeError("trainer mesh identity changed during plan preparation")

    monkeypatch.setattr(harness.transfer, "_verify_trainer_mesh", verify_mesh)
    with pytest.raises(RuntimeError, match="mesh identity changed"):
        harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    recovered = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 2, ()))
    assert recovered.metrics["plan_cache_hits"] == 0
    _check_values(harness, harness.collect(recovered)[1])
    for tensor in harness.sources.values():
        tensor.add_(4)
    reused = prepare_warm(harness)
    assert reused.metrics["plan_cache_hits"] == 1
    _check_values(harness, harness.collect(reused)[1])


def test_debug_layout_compares_tensor_index_arguments_before_transfer(
    harness, monkeypatch
) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")
    monkeypatch.setenv("MX_REFIT_DEBUG_VALIDATE_GENERATOR_LAYOUT", "1")
    indices = torch.tensor([0, 2])
    copy = harness.capture.copies[1]
    copy.op_chain = (("__getitem__", (indices,), ()),)
    copy.dest_shape = (2, 4)
    copy.dest_stride = (4, 1)
    first = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    _, installed = harness.collect(first)
    expected = torch.zeros(20, dtype=torch.float32)
    expected[2:10].copy_(harness.sources["full"][[0, 2]].reshape(-1))
    assert torch.equal(installed["b.weight"], expected)
    _, installed = harness.collect(harness.prepare(trainer_snapshot=harness.transfer.cached_trainer_source()))
    assert torch.equal(installed["b.weight"], expected)
    posts = harness.events.count("post")
    indices[0] = 1
    with pytest.raises(ValueError, match="generator layout changed"):
        harness.prepare(trainer_snapshot=harness.transfer.cached_trainer_source())
    assert harness.events.count("post") == posts


def test_manifest_parser_rows_are_owned_before_transfer(harness, monkeypatch) -> None:
    import modelexpress_rl.inference.nixl_staged_transfer as transfer_module

    parsed = []
    resolve = transfer_module._resolve_sources

    def observe(*args, **kwargs) -> transfer_module._ResolvedSources:
        result = resolve(*args, **kwargs)
        parsed.extend(result.sources.values())
        return result

    monkeypatch.setattr(transfer_module, "_resolve_sources", observe)
    prepared = harness.prepare()
    for source in parsed:
        source.global_shape = (1,)
        for shard in source.shards:
            shard.addr = 0
            shard.shape = (1,)
    _check_values(harness, harness.collect(prepared)[1])
    _check_values(harness, harness.collect(prepare_warm(harness))[1])

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
    monkeypatch.delenv("MX_REFIT_CACHE_PLAN")
    harness.new_transfer()
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
def test_debug_validation_accepts_changed_values_with_stable_structure(
    harness, monkeypatch, plan_debug
) -> None:
    monkeypatch.setenv("MX_REFIT_DEBUG_VALIDATE_PLAN", str(int(plan_debug)))
    harness.new_transfer()
    first = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    harness.collect(first)
    for tensor in harness.sources.values():
        tensor.add_(3)
    prepared = harness.prepare(
        trainer_snapshot=harness.transfer.cached_trainer_source()
    )
    assert prepared.metrics["plan_cache_hits"] == 1
    _check_values(harness, harness.collect(prepared)[1])



def test_debug_source_drift_fails_and_next_attempt_rebuilds(
    harness, monkeypatch
) -> None:
    monkeypatch.setenv("MX_REFIT_DEBUG_VALIDATE_PLAN", "1")
    harness.new_transfer()
    first = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    harness.collect(first)
    harness.sources["exact"] = harness.sources["exact"].clone().add_(5)
    with pytest.raises(ValueError, match="source metadata changed"):
        harness.prepare(trainer_snapshot=harness.transfer.cached_trainer_source())
    rebuilt = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    assert rebuilt.metrics["plan_cache_hits"] == 0
    _check_values(harness, harness.collect(rebuilt)[1])


@pytest.mark.parametrize(
    "flag", ["MX_REFIT_CACHE_PLAN"]
)
def test_explicit_cache_disable_requires_validated_preparation(
    harness, monkeypatch, flag
) -> None:
    monkeypatch.setenv(flag, "0")
    harness.new_transfer()
    first = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    harness.collect(first)
    prepared = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    assert prepared.metrics["plan_cache_hits"] == 0
    _check_values(harness, harness.collect(prepared)[1])


def test_replacing_candidate_under_same_mesh_rebuilds(harness) -> None:
    first = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    harness.collect(first)
    replacement = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    assert replacement.metrics["plan_cache_hits"] == 0
    _check_values(harness, harness.collect(replacement)[1])


def test_mesh_change_during_setup_rejects_then_rebuilds(harness) -> None:
    from types import SimpleNamespace

    from modelexpress_rl import refit_pb2
    from modelexpress_rl.inference.runtime import _create_load_time_tensor_method

    current = refit_pb2.TrainerMesh(mesh_id="mesh", generation=2)
    service = SimpleNamespace(
        GetTrainerMesh=lambda *args, **kwargs: refit_pb2.GetTrainerMeshResponse(
            mesh=current
        )
    )
    method = _create_load_time_tensor_method(
        capability=SimpleNamespace(
            device_id=0,
            device=torch.device("cuda:0"),
            capture_layout=harness.capture_layout,
        ),
        worker_id="target",
        service=lambda: service,
    )
    harness.transfer.close()
    # Use the runtime-created transfer so the real mesh RPC check runs during setup.
    harness.use_transfer(method._transfer)
    with pytest.raises(RuntimeError, match="mesh identity changed"):
        harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    recovered = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 2, ()))
    _check_values(harness, harness.collect(recovered)[1])
    for tensor in harness.sources.values():
        tensor.add_(4)
    _check_values(harness, harness.collect(prepare_warm(harness))[1])



def test_freeze_sources_isolates_caller_owned_rows(harness) -> None:
    from modelexpress_rl.inference._source_snapshot import _freeze_sources
    from modelexpress_rl.inference.nixl_staged_transfer import _resolve_sources

    parsed = _resolve_sources(harness.manifests()).sources
    owned = _freeze_sources(parsed)
    address = harness.sources["exact"].data_ptr()
    for source in parsed.values():
        source.global_shape = (1,)
        for shard in source.shards:
            shard.addr = 0
            shard.shape = (1,)
    assert owned["exact"].global_shape == (4, 4)
    assert owned["exact"].shards[0].addr == address
    assert owned["exact"].shards[0].shape == (4, 4)








@pytest.mark.parametrize(
    "buffers,failed_call", [(1, 1), (2, 1), (2, 2), (0, 1), (0, 2), (0, 3)]
)
def test_registration_failure_retries_with_fully_bound_workspace(
    harness, monkeypatch, buffers, failed_call
) -> None:
    from contextlib import nullcontext

    import modelexpress_rl.inference.nixl_staged_transfer as transfer_module

    original_empty = torch.empty

    def empty(shape, **kwargs) -> torch.Tensor:
        kwargs.pop("device", None)
        return original_empty(shape, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(transfer_module, "classic_cuda_alloc", nullcontext)
    harness.remote["fail_registration"] = failed_call

    def prepare() -> (
        transfer_module._PreparedBoundedTransfer | transfer_module._PreparedNixlTransfer
    ):
        if buffers:
            return harness.prepare(staging_buffers=buffers)
        return harness.transfer.prepare_full_copy(
            manifests=harness.manifests(),
            trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()),
            capture_layout=harness.capture_layout,
        )

    with pytest.raises(RuntimeError, match="registration failed"):
        prepare()
    assert "post" not in harness.events
    harness.remote["fail_registration"] = None
    prepared = prepare()
    if buffers:
        _check_values(harness, harness.collect(prepared)[1])
    else:
        tensors = harness.transfer.stage(prepared).tensors
        for source, name, transpose in (
            ("exact", "a.weight", False),
            ("full", "b.weight", True),
            ("convert", "c.weight", False),
            ("exact", "d.weight", False),
        ):
            value = harness.sources[source]
            expected = (
                (value.T if transpose else value).reshape(-1).to(tensors[name].dtype)
            )
            assert torch.equal(tensors[name][2:18], expected)



def test_digest_policy_is_fixed_for_the_transfer_lifetime(harness, monkeypatch) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    harness.new_transfer()
    _check_values(harness, harness.collect(harness.prepare())[1])
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")

    prepared = harness.prepare()
    for source in harness.sources.values():
        source.add_(2)
    _check_values(harness, harness.collect(prepared)[1])
    harness.new_transfer()
    prepared = harness.prepare()
    harness.sources["exact"].add_(1)
    with pytest.raises(RuntimeError, match="digest"):
        harness.collect(prepared)


def test_digest_warm_bindings_use_current_manifest_digests(
    harness, monkeypatch
) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")
    harness.new_transfer()
    _check_values(harness, harness.collect(harness.prepare())[1])
    for source in harness.sources.values():
        source.add_(3)
    prepared = harness.prepare()
    _check_values(harness, harness.collect(prepared)[1])
    prepared = harness.prepare()
    harness.sources["exact"].add_(1)
    with pytest.raises(RuntimeError, match="digest"):
        harness.collect(prepared)


@pytest.mark.parametrize("setting", ["max_staging_bytes", "staging_buffers"])
@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_streaming_prepare_rejects_invalid_budget_settings_and_recovers(
    harness, setting, invalid
) -> None:
    settings = {"max_staging_bytes": 1024, "staging_buffers": 1}
    settings[setting] = invalid
    with pytest.raises(ValueError, match=f"{setting} must be a positive integer"):
        harness.transfer.prepare_streaming(
            manifests=harness.manifests(),
            trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()),
            capture_layout=harness.capture_layout,
            staging_device="cpu",
            **settings,
        )
    assert not harness.transports
    prepared = harness.prepare()
    _check_values(harness, harness.collect(prepared)[1])


def test_closed_transfer_rejects_streaming_preparation(harness) -> None:
    harness.transfer.close()
    with pytest.raises(RuntimeError, match="closed"):
        harness.prepare()
    assert not harness.transports

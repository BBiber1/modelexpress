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
    monkeypatch.delenv("MX_REFIT_CACHE_GENERATOR_LAYOUT")
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
@pytest.mark.parametrize("layout_debug", [False, True])
def test_debug_validation_accepts_changed_values_with_stable_structure(
    harness, monkeypatch, plan_debug, layout_debug
) -> None:
    monkeypatch.setenv("MX_REFIT_DEBUG_VALIDATE_PLAN", str(int(plan_debug)))
    monkeypatch.setenv(
        "MX_REFIT_DEBUG_VALIDATE_GENERATOR_LAYOUT", str(int(layout_debug))
    )
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


def test_debug_layout_drift_fails_before_transfer(harness, monkeypatch) -> None:
    monkeypatch.setenv("MX_REFIT_DEBUG_VALIDATE_GENERATOR_LAYOUT", "1")
    harness.new_transfer()
    first = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    harness.collect(first)
    harness.capture.copies[0].dest_offset += 1
    with pytest.raises(ValueError, match="generator layout changed"):
        harness.prepare(trainer_snapshot=harness.transfer.cached_trainer_source())


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
    "flag", ["MX_REFIT_CACHE_GENERATOR_LAYOUT", "MX_REFIT_CACHE_PLAN"]
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
    harness.new_transfer()
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
    _, installed = harness.collect(
        harness.prepare(trainer_snapshot=harness.transfer.cached_trainer_source())
    )
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


@pytest.mark.parametrize("plan_cache", [False, True])
@pytest.mark.parametrize("layout_cache", [False, True])
@pytest.mark.parametrize("descriptor_cache", [False, True])
def test_plan_and_layout_controls_have_independent_lifetimes(
    harness, monkeypatch, plan_cache, layout_cache, descriptor_cache
) -> None:
    import modelexpress_rl.inference.nixl_staged_transfer as transfer_module

    monkeypatch.setenv("MX_REFIT_CACHE_PLAN", str(int(plan_cache)))
    monkeypatch.setenv("MX_REFIT_CACHE_GENERATOR_LAYOUT", str(int(layout_cache)))
    monkeypatch.setenv("MX_REFIT_CACHE_DESCRIPTORS", str(int(descriptor_cache)))
    harness.new_transfer()
    builds = []
    compile_plan = transfer_module._plan_staged_transfer

    def observe(*args, **kwargs) -> transfer_module.TransferPlan:
        builds.append(1)
        return compile_plan(*args, **kwargs)

    monkeypatch.setattr(transfer_module, "_plan_staged_transfer", observe)
    first = harness.prepare()
    _check_values(harness, harness.collect(first)[1])
    first_builds = len(builds)
    first_descriptors = first.metrics["descriptor_builds"]
    for tensor in harness.sources.values():
        tensor.add_(3)
    second = harness.prepare()
    _check_values(harness, harness.collect(second)[1])
    assert len(harness.captures) == (1 if layout_cache else 2)
    assert len(builds) == (
        first_builds if plan_cache and layout_cache else 2 * first_builds
    )
    assert second.metrics["descriptor_builds"] == (
        0 if plan_cache and layout_cache and descriptor_cache else first_descriptors
    )
    assert len(harness.allocations) == 1


def test_mesh_address_change_preserves_capture_and_registered_arena(harness) -> None:
    first = harness.prepare()
    _check_values(harness, harness.collect(first)[1])
    harness.sources["exact"] = harness.sources["exact"].clone().add_(7)
    second = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 2, ()))
    _check_values(harness, harness.collect(second)[1])
    assert len(harness.captures) == 1
    assert len(harness.allocations) == 1
    assert second.metrics["plan_cache_misses"] == 1


@pytest.mark.parametrize("remove_ok", [False, True])
def test_mesh_agent_replacement_rebinds_after_safe_removal(harness, remove_ok) -> None:
    first = harness.prepare()
    _check_values(harness, harness.collect(first)[1])
    harness.remote.update(name="replacement", remove_ok=remove_ok)
    for tensor in harness.sources.values():
        tensor.add_(4)
    replacement = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 2, ()))
    _check_values(harness, harness.collect(replacement)[1])
    assert harness.events.count("remove:source") == 1
    assert len(harness.allocations) == (1 if remove_ok else 2)
    assert len(harness.captures) == 1
    _check_values(harness, harness.collect(prepare_warm(harness))[1])


@pytest.mark.parametrize("bounded", [False, True])
def test_changed_source_schema_recaptures_and_rebinds_larger_storage(
    harness, monkeypatch, bounded
) -> None:
    from contextlib import nullcontext

    import modelexpress_rl.inference.nixl_staged_transfer as transfer_module

    original_empty = torch.empty

    def empty(shape, **kwargs) -> torch.Tensor:
        kwargs.pop("device", None)
        return original_empty(shape, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(transfer_module, "classic_cuda_alloc", nullcontext)

    def prepare(
        generation,
    ) -> (
        transfer_module._PreparedBoundedTransfer | transfer_module._PreparedNixlTransfer
    ):
        if bounded:
            return harness.prepare(
                trainer_snapshot=TrainerSourceSnapshot("mesh", generation, ())
            )
        return harness.transfer.prepare_full_copy(
            manifests=harness.manifests(),
            trainer_snapshot=TrainerSourceSnapshot("mesh", generation, ()),
            capture_layout=harness.capture_layout,
        )

    def install(prepared) -> dict[str, torch.Tensor]:
        if bounded:
            return harness.collect(prepared)[1]
        return harness.transfer.stage(prepared).tensors

    initial = install(prepare(1))
    if bounded:
        _check_values(harness, initial)
    else:
        assert torch.equal(
            initial["a.weight"][2:18], harness.sources["exact"].reshape(-1)
        )
    harness.sources["exact"] = torch.arange(128, dtype=torch.float32).reshape(32, 4)
    for copy in (harness.capture.copies[0], harness.capture.copies[3]):
        copy.dest_shape = (32, 4)
        copy.dest_stride = (4, 1)
        harness.layout[copy.param_name] = ((132,), torch.float32)
    values = install(prepare(2))
    expected = torch.zeros(132)
    expected[2:130].copy_(harness.sources["exact"].reshape(-1))
    for name in ("a.weight", "d.weight"):
        assert torch.equal(values[name][2:130], expected[2:130])
        if bounded:
            assert torch.equal(values[name], expected)
    assert len(harness.captures) == 2
    if bounded:
        assert len(harness.allocations) == 2


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


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("descriptors", [False, True])
def test_warm_execution_only_rebinds_when_descriptors_disabled(
    harness, monkeypatch, bounded, descriptors
) -> None:
    from contextlib import nullcontext

    import modelexpress_rl.inference.nixl_staged_transfer as module

    monkeypatch.setenv("MX_REFIT_CACHE_DESCRIPTORS", str(int(descriptors)))
    harness.new_transfer()
    original_empty = torch.empty

    def empty(shape, **kwargs) -> torch.Tensor:
        kwargs.pop("device", None)
        return original_empty(shape, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(module, "classic_cuda_alloc", nullcontext)

    def prepare() -> module._PreparedBoundedTransfer | module._PreparedNixlTransfer:
        if bounded:
            return harness.prepare()
        return harness.transfer.prepare_full_copy(
            manifests=harness.manifests(),
            trainer_snapshot=harness.transfer.cached_trainer_source()
            or TrainerSourceSnapshot("mesh", 1, ()),
            capture_layout=harness.capture_layout,
        )

    def install(prepared) -> dict[str, torch.Tensor]:
        if bounded:
            return harness.collect(prepared)[1]
        return harness.transfer.stage(prepared).tensors

    install(prepare())
    builds = []
    original_descriptors = harness.transfer._descriptors

    def bind(*args, **kwargs) -> list[module.ReadDescriptor]:
        builds.append(1)
        return original_descriptors(*args, **kwargs)

    def unexpected(*args, **kwargs) -> None:
        raise AssertionError(
            "normal warm execution performed diagnostic or planning work"
        )

    monkeypatch.setattr(harness.transfer, "_descriptors", bind)
    monkeypatch.setattr(module, "_arena_geometry", unexpected)
    monkeypatch.setattr(harness.transfer, "_resolve_metadata", unexpected)
    monkeypatch.setattr(harness.transfer, "_resolve_layout", unexpected)
    monkeypatch.setattr(harness.transfer, "_validate_complete", unexpected)
    for source in harness.sources.values():
        source.add_(5)
    prepared = prepare()
    installed = install(prepared)
    assert torch.equal(
        installed["a.weight"][2:18], harness.sources["exact"].reshape(-1)
    )
    assert len(builds) == (0 if descriptors else prepared.metrics.get("batches", 1))
    assert prepared.metrics["descriptor_builds"] == len(builds)


def test_digest_policy_is_fixed_for_the_transfer_lifetime(harness, monkeypatch) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    harness.new_transfer()
    _check_values(harness, harness.collect(harness.prepare())[1])
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")

    def unexpected(*args, **kwargs) -> None:
        raise AssertionError("a live environment change enabled digest verification")

    monkeypatch.setattr(harness.transfer, "_verify", unexpected)
    for source in harness.sources.values():
        source.add_(2)
    _check_values(harness, harness.collect(harness.prepare())[1])


def test_digest_warm_bindings_use_current_manifest_digests(
    harness, monkeypatch
) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")
    harness.new_transfer()
    _check_values(harness, harness.collect(harness.prepare())[1])
    for source in harness.sources.values():
        source.add_(3)
    prepared = harness.prepare()
    assert prepared.metrics["descriptor_builds"] == 0
    _check_values(harness, harness.collect(prepared)[1])
    prepared = harness.prepare()
    harness.sources["exact"].add_(1)
    with pytest.raises(RuntimeError, match="digest"):
        harness.collect(prepared)

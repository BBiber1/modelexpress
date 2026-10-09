# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from modelexpress import telemetry
from modelexpress.refit.timing import RefitTimingRecorder, use_refit_timing
from modelexpress_rl.inference.plan import TrainerSourceSnapshot
from modelexpress_rl.inference.source.trainer import TrainerSourceResolver
from opentelemetry import trace

from tests import test_nixl_telemetry, test_refit_bounded_descriptors
from tests.test_refit_trainer_source_resolution import sources

recording = test_nixl_telemetry.recording
harness = test_refit_bounded_descriptors.harness


@pytest.mark.parametrize("count", [1, 256])
def test_cold_shards_emit_one_resolution_span_closed_before_yield(
    recording, monkeypatch, count
) -> None:
    resolver, version, fetched, size, service = sources(monkeypatch, count)
    recorder = RefitTimingRecorder(backend="rl_generator", version="v1", rank=0)
    with (
        use_refit_timing(recorder),
        telemetry.span("mx.refit.streaming_prepare") as parent,
    ):
        candidates = resolver.candidates(version)
        candidate = next(candidates)
        assert len(candidate.shards) == count
        assert trace.get_current_span() is parent
        spans = recording.get_finished_spans()
        resolution = [s for s in spans if s.name == "mx.refit.source_resolution"]
        assert len(resolution) == 1
        assert resolution[0].attributes["manifest_fetch_count"] == count
        assert resolution[0].attributes["manifest_bytes"] == count * size
        assert not any(s.name == "mx.refit.source_preparation" for s in spans)
        candidates.close()
    assert len(fetched) == count


def test_mixed_shards_keep_fetch_diagnostics_and_count_failed_replicas(
    recording, monkeypatch
) -> None:
    resolver, version, fetched, size, service = sources(
        monkeypatch, 3, failed_replica=True
    )
    recorder = RefitTimingRecorder(backend="rl_generator", version="v1", rank=0)
    with use_refit_timing(recorder), telemetry.span("mx.refit.streaming_prepare"):
        candidates = resolver.candidates(version)
        candidate = next(candidates)
        assert len(candidate.shards) == 3
        candidates.close()
    spans = recording.get_finished_spans()
    resolution = next(s for s in spans if s.name == "mx.refit.source_resolution")
    assert resolution.attributes["manifest_fetch_count"] == 4
    assert resolution.attributes["manifest_fetch_bytes"] == 3 * size
    assert resolution.attributes["manifest_bytes"] == 3 * size
    assert fetched == ["a-failed", "worker-0", "worker-1", "worker-2"]
    fetches = [s for s in spans if s.name == "mx.refit.manifest_fetch"]
    assert len(fetches) == 4
    assert sum(s.attributes.get("status") == "error" for s in fetches) == 1
    assert any(s.name == "mx.refit.manifest_hash" for s in spans)
    assert any(s.name == "mx.refit.manifest_fingerprint" for s in spans)


def test_staging_phases_are_connected_and_warm_arenas_are_reused(
    recording, harness
) -> None:
    with telemetry.span("mx.refit.streaming_prepare") as parent:
        harness.prepare()
    spans = recording.get_finished_spans()
    names = {s.name for s in spans}
    expected = {
        "source_metadata",
        "layout_capture",
        "transfer_planning",
        "required_agents",
        "connection_registration",
        "receive_workspace",
        "arena_allocation",
        "arena_registration",
        "descriptor_preparation",
    }
    assert {"mx.refit." + name for name in expected} <= names
    assert all(s.context.trace_id == parent.get_span_context().trace_id for s in spans)
    recording.clear()
    with telemetry.span("mx.refit.streaming_prepare"):
        harness.prepare()
    names = {s.name for s in recording.get_finished_spans()}
    assert "mx.refit.required_agents" not in names
    assert "mx.refit.arena_allocation" not in names
    assert "mx.refit.arena_registration" not in names


def test_disabled_telemetry_keeps_candidate_resolution_working(monkeypatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    resolver, version, fetched, _, service = sources(monkeypatch, 256)
    assert len(next(resolver.candidates(version)).shards) == 256
    assert len(fetched) == 256


def test_mesh_warm_trace_reports_lookup_without_cold_work(
    recording, harness, monkeypatch
) -> None:
    cold = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    harness.collect(cold)
    original, version, fetched, _, service = sources(monkeypatch, 256)

    def unexpected_listing(*args, **kwargs) -> None:
        pytest.fail("warm refit listed published shards")

    service.ListWeightVersionShards = unexpected_listing
    resolver = TrainerSourceResolver(
        service=lambda: service,
        rpc_timeout_seconds=1,
        cached_source=harness.transfer.cached_trainer_source,
    )
    recording.clear()
    recorder = RefitTimingRecorder(backend="rl_generator", version="v2", rank=0)
    for tensor in harness.sources.values():
        tensor.add_(2)
    with (
        use_refit_timing(recorder),
        telemetry.span("mx.refit.streaming_prepare") as parent,
    ):
        candidates = resolver.candidates(version)
        candidate = next(candidates)
        prepared = harness.transfer.prepare_streaming(
            manifests=None,
            trainer_snapshot=candidate,
            capture_layout=lambda _: (harness.capture, harness.layout),
            max_staging_bytes=1024,
            staging_device="cpu",
        )
        candidates.close()
    test_refit_bounded_descriptors._check_values(harness, harness.collect(prepared)[1])
    names = {span.name for span in recording.get_finished_spans()}
    assert {"mx.refit.mesh_lookup", "mx.refit.plan_cache_lookup"} <= names
    assert not names & {
        "mx.refit.source_metadata",
        "mx.refit.layout_capture",
        "mx.refit.transfer_planning",
        "mx.refit.required_agents",
        "mx.refit.manifest_fetch",
        "mx.refit.descriptor_preparation",
        "mx.refit.workspace_validate",
        "mx.refit.plan_cache_validate",
    }
    assert parent.attributes["mx.measurement.plan_cache_hits"] == 1
    assert parent.attributes["mx.measurement.descriptor_builds"] == 0
    assert fetched == []


def test_same_mesh_replacement_trace_reports_cache_miss(recording, harness) -> None:
    first = harness.prepare(trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ()))
    harness.collect(first)
    for tensor in harness.sources.values():
        tensor.add_(3)
    recording.clear()
    with telemetry.span("mx.refit.streaming_prepare"):
        replaced = harness.prepare(
            trainer_snapshot=TrainerSourceSnapshot("mesh", 1, ())
        )
    test_refit_bounded_descriptors._check_values(harness, harness.collect(replaced)[1])
    spans = recording.get_finished_spans()
    lookup = next(span for span in spans if span.name == "mx.refit.plan_cache_lookup")
    assert lookup.attributes["mx.refit.plan_cache_hits"] == 0
    assert any(span.name == "mx.refit.transfer_planning" for span in spans)


def test_mode_switch_lookup_reports_miss_and_performs_cold_work(
    recording, harness
) -> None:
    with telemetry.span("mx.refit.streaming_prepare"):
        for bounded in (False, True):
            for _ in range(2):
                for tensor in harness.sources.values():
                    tensor.add_(2)
                if bounded:
                    prepared = harness.prepare()
                    test_refit_bounded_descriptors._check_values(
                        harness, harness.collect(prepared)[1]
                    )
                else:
                    prepared = harness.transfer.prepare_full_copy(
                        manifests=harness.manifests(),
                        trainer_snapshot=harness.transfer.cached_trainer_source()
                        or TrainerSourceSnapshot("mesh", 1, ()),
                        capture_layout=harness.capture_layout,
                    )
                    installed = harness.transfer.stage(prepared).tensors
                    for source, name, transpose in (
                        ("exact", "a.weight", False),
                        ("full", "b.weight", True),
                        ("convert", "c.weight", False),
                        ("exact", "d.weight", False),
                    ):
                        value = harness.sources[source]
                        expected = (value.T if transpose else value).reshape(-1)
                        assert torch.equal(
                            installed[name][2:18], expected.to(installed[name].dtype)
                        )
    spans = recording.get_finished_spans()
    lookups = [
        span.attributes["mx.refit.plan_cache_hits"]
        for span in spans
        if span.name == "mx.refit.plan_cache_lookup"
    ]
    assert lookups == [0, 1, 0, 1]
    assert sum(span.name == "mx.refit.transfer_planning" for span in spans) == 2


def test_debug_checks_are_exported_only_when_enabled(
    recording, harness, monkeypatch
) -> None:
    for flag in (
        "MX_REFIT_DEBUG_VALIDATE_PLAN",
        "MX_REFIT_DEBUG_VALIDATE_GENERATOR_LAYOUT",
        "MX_REFIT_DEBUG_VALIDATE_WORKSPACE",
    ):
        monkeypatch.setenv(flag, "1")
    harness.new_transfer()
    harness.collect(harness.prepare())
    recording.clear()
    for tensor in harness.sources.values():
        tensor.add_(2)
    with telemetry.span("mx.refit.streaming_prepare"):
        prepared = harness.prepare()
        _, installed = harness.collect(prepared)
    test_refit_bounded_descriptors._check_values(harness, installed)
    names = {span.name for span in recording.get_finished_spans()}
    assert {
        "mx.refit.source_metadata",
        "mx.refit.layout_capture",
        "mx.refit.plan_cache_validate",
        "mx.refit.workspace_validate",
    } <= names
    assert "mx.refit.descriptor_preparation" not in names


def test_reconstruction_spans_include_completion_fences(
    recording, harness, monkeypatch
) -> None:
    import torch

    prepared = harness.prepare()
    recording.clear()
    synchronized = set()
    original_sync = torch.cuda.synchronize

    def synchronize(*args, **kwargs) -> None:
        span = trace.get_current_span()
        if getattr(span, "name", None) == "mx.refit.reconstruct":
            synchronized.add(span.get_span_context().span_id)
        original_sync(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    with telemetry.span("mx.refit.receive"):
        _, installed = harness.collect(prepared)
    test_refit_bounded_descriptors._check_values(harness, installed)
    reconstruction = [
        span
        for span in recording.get_finished_spans()
        if span.name == "mx.refit.reconstruct"
    ]
    assert len(reconstruction) == prepared.metrics["batches"]
    assert {span.context.span_id for span in reconstruction} == synchronized


def test_workspace_validation_span_reports_drift_before_reads(
    recording, harness, monkeypatch
) -> None:
    monkeypatch.setenv("MX_REFIT_DEBUG_VALIDATE_WORKSPACE", "1")
    harness.new_transfer()
    harness.collect(harness.prepare())
    arena = harness.allocated_tensors[0]
    arena.resize_(arena.numel() + 256)
    posts = harness.events.count("post")
    recording.clear()
    with pytest.raises(ValueError, match="registered workspace changed"):
        with telemetry.span("mx.refit.streaming_prepare"):
            harness.prepare()
    checks = [
        span
        for span in recording.get_finished_spans()
        if span.name == "mx.refit.workspace_validate"
    ]
    assert len(checks) == 1
    assert checks[0].status.status_code is trace.StatusCode.ERROR
    assert harness.events.count("post") == posts

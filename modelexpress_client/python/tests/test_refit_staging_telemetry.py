# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from types import SimpleNamespace
from collections.abc import Iterator

import pytest
from opentelemetry import trace

from modelexpress import telemetry
from modelexpress.refit.timing import use_refit_timing
from modelexpress_rl import refit_pb2
from modelexpress_rl.inference.plan import StreamingSettings
import modelexpress_rl.inference.source.trainer as trainer_source_module
from tests import test_nixl_telemetry, test_refit_bounded_descriptors
from tests.test_refit_shard_metadata import (
    _advertise,
    _next_version,
    _remember,
    discovery_timing,
    protocol,
)

recording = test_nixl_telemetry.recording
harness = test_refit_bounded_descriptors.harness


def _resolution_spans(recording) -> list:
    return [span for span in recording.get_finished_spans()
            if span.name == "mx.refit.source_resolution"]


@pytest.fixture
def exported_metrics(recording, monkeypatch: pytest.MonkeyPatch) -> Iterator:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://unused/v1/metrics")
    monkeypatch.setattr(telemetry, "_meter_provider", provider)
    telemetry._value_histogram.cache_clear()
    telemetry._histogram.cache_clear()
    try:
        yield reader
    finally:
        provider.shutdown()
        telemetry._value_histogram.cache_clear()
        telemetry._histogram.cache_clear()


def _assert_exported_counters(reader, stage: dict) -> None:
    exported = {}
    data = reader.get_metrics_data()
    if data is not None:
        for resource in data.resource_metrics:
            for scope in resource.scope_metrics:
                for metric in scope.metrics:
                    exported[metric.name] = sum(point.sum for point in metric.data.data_points)
    for key, value in stage["metadata"].items():
        if isinstance(value, (int, float)) and not key.endswith("_s"):
            assert exported.get("mx_refit_" + key, 0) == pytest.approx(value), key


@pytest.mark.parametrize("verify", [False, True])
def test_cold_and_warm_resolution_export_counters_once_before_yield(
    recording, exported_metrics, protocol: SimpleNamespace, discovery_timing: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch, verify: bool,
) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", str(int(verify)))
    with use_refit_timing(discovery_timing.recorder), telemetry.span("mx.refit.streaming_prepare") as parent:
        candidates = protocol.resolver.candidates(protocol.version)
        first = next(candidates)
        assert trace.get_current_span() is parent
        assert len(_resolution_spans(recording)) == 1
        candidates.close()
        _remember(protocol, first)
        candidates = protocol.resolver.candidates(_next_version(protocol))
        second = next(candidates)
        assert trace.get_current_span() is parent
        assert len(_resolution_spans(recording)) == (2 if verify else 1)
        discovery_timing.clock.now += 100
        candidates.close()
    stage = discovery_timing.recorder.as_dict()["stages"]["source_preparation"]
    _assert_exported_counters(exported_metrics, stage)
    assert protocol.control.list_calls == 2
    assert len(protocol.worker.stable_requests) == 1
    assert len(protocol.worker.version_requests) == (2 if verify else 0)
    assert list(second.checksums.values()) == (["second"] if verify else [])
    assert stage["duration_ms"] == (8000 if verify else 2000)
    resolutions = _resolution_spans(recording)
    assert sum(span.attributes.get("manifest_fetch_count", 0) for span in resolutions) == 1
    assert sum(span.attributes.get("version_metadata_fetch_count", 0) for span in resolutions) == (2 if verify else 0)
    assert parent.attributes["mx.measurement.manifest_cache_hits"] == 1
    for span in resolutions:
        assert span.end_time >= span.start_time
        assert span.parent.span_id == parent.get_span_context().span_id
        assert span.context.trace_id == parent.get_span_context().trace_id
    assert not any(span.name == "mx.refit.source_preparation" for span in recording.get_finished_spans())


@pytest.mark.parametrize("failure", ["stable", "version"])
def test_failed_replica_fallback_exports_partial_work_once(
    recording, exported_metrics, protocol: SimpleNamespace, discovery_timing: SimpleNamespace, failure: str,
) -> None:
    alternate = refit_pb2.WeightVersionShard()
    alternate.CopyFrom(protocol.control.shards[0])
    alternate.worker_id = "trainer-b"
    protocol.worker.metadata.worker_id = "trainer-b"
    _advertise(protocol)
    alternate.version_metadata_digest = protocol.control.shards[0].version_metadata_digest
    protocol.control.shards.append(alternate)
    setattr(protocol.worker, f"{failure}_failures", 1)
    with use_refit_timing(discovery_timing.recorder), telemetry.span("mx.refit.streaming_prepare") as parent:
        candidates = protocol.resolver.candidates(protocol.version)
        resolved = next(candidates)
        assert trace.get_current_span() is parent
        assert resolved.snapshot.shards[0].worker_id == "trainer-b"
        assert len(_resolution_spans(recording)) == 1
        candidates.close()
    stage = discovery_timing.recorder.as_dict()["stages"]["source_preparation"]
    _assert_exported_counters(exported_metrics, stage)
    assert len(protocol.worker.stable_requests) == 2
    assert len(protocol.worker.version_requests) == (1 if failure == "stable" else 2)
    assert stage["metadata"]["manifest_fetch_count"] == (1 if failure == "stable" else 2)
    assert stage["metadata"]["manifest_bytes"] == len(protocol.worker.blob)
    assert stage["duration_ms"] == (7000 if failure == "stable" else 10000)
    resolution, = _resolution_spans(recording)
    assert resolution.attributes["status"] == "error"


@pytest.mark.parametrize("resume", [False, True])
def test_lazy_resolution_restores_context_and_excludes_consumer_pause(
    recording, exported_metrics, protocol: SimpleNamespace, discovery_timing: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch, resume: bool,
) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    monkeypatch.setattr(trainer_source_module, "time_ns", lambda: int(discovery_timing.clock.now * 1_000_000_000))
    alternate = refit_pb2.WeightVersionShard()
    alternate.CopyFrom(protocol.control.shards[0])
    alternate.worker_id = "trainer-b"
    payload = json.loads(protocol.worker.blob)
    payload["tensors"][0]["shards"][0]["addr"] = 8192
    blob = json.dumps(payload).encode()
    alternate.stable_metadata_digest = hashlib.sha256(blob).hexdigest()
    protocol.worker.blobs[alternate.stable_metadata_digest] = blob
    protocol.control.shards.append(alternate)
    with use_refit_timing(discovery_timing.recorder), telemetry.span("mx.refit.streaming_prepare") as parent:
        candidates = protocol.resolver.candidates(protocol.version)
        assert next(candidates).snapshot.shards[0].worker_id == "trainer-0"
        assert trace.get_current_span() is parent
        with telemetry.span("consumer_pause") as consumer:
            discovery_timing.clock.now += 100
            assert trace.get_current_span() is consumer
        if resume:
            assert next(candidates).snapshot.shards[0].worker_id == "trainer-b"
        assert trace.get_current_span() is parent
        before_close = len(recording.get_finished_spans())
        discovery_timing.clock.now += 100
        candidates.close()
        assert len(recording.get_finished_spans()) == before_close
        assert trace.get_current_span() is parent
    stage = discovery_timing.recorder.as_dict()["stages"]["source_preparation"]
    _assert_exported_counters(exported_metrics, stage)
    count = 2 if resume else 1
    resolutions = _resolution_spans(recording)
    assert len(resolutions) == count
    assert [span.end_time - span.start_time for span in resolutions] == [2_000_000_000] * count
    if resume:
        assert resolutions[1].start_time - resolutions[0].end_time == 100_000_000_000
    assert protocol.control.list_calls == 1
    assert len(protocol.worker.stable_requests) == count
    assert stage["duration_ms"] == 2000 * count
    consumer, = [span for span in recording.get_finished_spans() if span.name == "consumer_pause"]
    assert consumer.parent.span_id == parent.get_span_context().span_id


def test_disabled_telemetry_preserves_current_protocol_resolution(
    protocol: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    resolved, = protocol.resolver.candidates(protocol.version)
    assert len(resolved.snapshot.shards) == 1
    assert list(resolved.checksums.values()) == ["first"]
    assert len(protocol.worker.stable_requests) == len(protocol.worker.version_requests) == 1


def test_staging_phase_parents_and_warm_registered_workspace_values(recording, harness) -> None:
    with telemetry.span("mx.refit.streaming_prepare") as parent:
        cold = harness.prepare()
    spans = recording.get_finished_spans()
    expected = {"source_metadata", "layout_capture", "bounded_plan_compile", "required_agents",
                "connection_registration", "receive_workspace", "arena_allocation",
                "arena_registration", "descriptor_preparation"}
    assert {"mx.refit." + name for name in expected} <= {span.name for span in spans}
    assert all(span.context.trace_id == parent.get_span_context().trace_id for span in spans)
    test_refit_bounded_descriptors._check_values(harness, harness.collect(cold)[1])
    registrations = harness.events.count("register")
    connections = sum(event.startswith("connect:") for event in harness.events)
    for tensor in harness.sources.values():
        tensor.add_(2)
    recording.clear()
    with telemetry.span("mx.refit.streaming_prepare"):
        warm = harness.prepare()
    metrics, installed = harness.collect(warm)
    test_refit_bounded_descriptors._check_values(harness, installed)
    assert harness.events.count("register") == registrations
    assert sum(event.startswith("connect:") for event in harness.events) == connections
    assert warm.metrics["plan_cache_hits"] == 1
    assert metrics["descriptor_builds"] == 0
    names = {span.name for span in recording.get_finished_spans()}
    assert "mx.refit.plan_cache_lookup" in names
    assert not names & {"mx.refit.layout_capture", "mx.refit.bounded_plan_compile",
                        "mx.refit.arena_allocation", "mx.refit.arena_registration"}


def test_mesh_generation_replacement_exports_cold_work_and_current_values(recording, harness) -> None:
    harness.collect(harness.prepare())
    for tensor in harness.sources.values():
        tensor.add_(3)
    recording.clear()
    with telemetry.span("mx.refit.streaming_prepare"):
        replacement = harness.prepare(generation=2)
    metrics, installed = harness.collect(replacement)
    test_refit_bounded_descriptors._check_values(harness, installed)
    assert replacement.metrics["plan_cache_hits"] == 0
    assert sum(span.name == "mx.refit.bounded_plan_compile" for span in recording.get_finished_spans()) == 1


@pytest.mark.parametrize("failure", ["stable", "version"])
def test_exhausted_resolution_exports_failed_rpc_time_and_restores_parent(
    recording, exported_metrics, protocol: SimpleNamespace, discovery_timing: SimpleNamespace, failure: str,
) -> None:
    setattr(protocol.worker, f"{failure}_failures", 1)
    with use_refit_timing(discovery_timing.recorder), telemetry.span("mx.refit.streaming_prepare") as parent:
        assert list(protocol.resolver.candidates(protocol.version)) == []
        assert trace.get_current_span() is parent
    stage = discovery_timing.recorder.as_dict()["stages"]["source_preparation"]
    _assert_exported_counters(exported_metrics, stage)
    assert len(_resolution_spans(recording)) == 1
    assert stage["metadata"].get("manifest_bytes", 0) == 0
    assert stage["duration_ms"] == (2000 if failure == "stable" else 5000)


@pytest.mark.parametrize("workspace_debug", [True])
def test_workspace_debug_span_rejects_drift_before_read(
    recording, harness, workspace_debug: bool,
) -> None:
    prepared = harness.prepare()
    # Change a registered arena to exercise the diagnostic at the READ boundary.
    arena = harness.transfer._staging_arenas[0]
    arena.resize_(arena.numel() + 256)
    posts = harness.events.count("post")
    recording.clear()
    with telemetry.span("mx.refit.receive"):
        with pytest.raises(RuntimeError, match="bounded workspace changed"):
            harness.collect(prepared)
    checks = [span for span in recording.get_finished_spans()
              if span.name == "mx.refit.workspace_validate"]
    assert len(checks) == 1
    assert checks[0].status.status_code is trace.StatusCode.ERROR
    assert harness.events.count("post") == posts


def test_reconstruction_spans_include_completion_fences(recording, harness, monkeypatch) -> None:
    import torch

    prepared = harness.prepare()
    recording.clear()
    synchronized = set()

    def synchronize(*args, **kwargs) -> None:
        span = trace.get_current_span()
        if getattr(span, "name", None) == "mx.refit.reconstruct":
            synchronized.add(span.get_span_context().span_id)

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    with telemetry.span("mx.refit.receive"):
        metrics, installed = harness.collect(prepared)
    test_refit_bounded_descriptors._check_values(harness, installed)
    reconstruction = [span for span in recording.get_finished_spans()
                      if span.name == "mx.refit.reconstruct"]
    assert len(reconstruction) == metrics["batches"]
    assert {span.context.span_id for span in reconstruction} == synchronized


@pytest.mark.parametrize("harness", [None, StreamingSettings(1024, "cpu")], indirect=True)
def test_fixed_mode_owner_exports_cold_miss_then_warm_hit_with_current_values(
    recording, harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import nullcontext

    import torch
    import modelexpress_rl.inference.nixl_staged_transfer as transfer_module

    monkeypatch.setattr(harness.transfer, "_device", torch.device("cpu"))
    monkeypatch.setattr(transfer_module, "classic_cuda_alloc", nullcontext)
    hits = []
    bounded = harness.streaming is not None
    with telemetry.span("mx.refit.streaming_prepare"):
        for _ in range(2):
            for tensor in harness.sources.values():
                tensor.add_(2)
            if bounded:
                prepared = harness.prepare()
                _, installed = harness.collect(prepared)
            else:
                snapshot = test_refit_bounded_descriptors._agent_registration_snapshot(
                    harness, generation=1, tokens={"source": b"source"},
                )
                cached = harness.transfer.cached_trainer_source()
                if cached is not None and cached.physical_fingerprint == snapshot.physical_fingerprint:
                    snapshot = cached
                prepared = harness.transfer.prepare_full_copy(
                    manifests=[shard.metadata for shard in snapshot.shards],
                    trainer_snapshot=snapshot, capture_layout=lambda _: (harness.capture, harness.layout),
                )
                installed = harness.transfer.stage(prepared).tensors
            test_refit_bounded_descriptors._check_values(harness, installed, check_padding=bounded)
            hits.append(prepared.metrics["plan_cache_hits"])
    assert hits == [0, 1]
    spans = recording.get_finished_spans()
    compile_name = "mx.refit.bounded_plan_compile" if bounded else "mx.refit.transfer_planning"
    assert sum(span.name == compile_name for span in spans) == 1
    if not bounded:
        assert sum(span.name == "mx.refit.transfer_validation" for span in spans) == 1


@pytest.mark.parametrize("count", [1, 256])
def test_many_cold_physical_shards_export_one_completed_resolution(
    recording, exported_metrics, protocol: SimpleNamespace, discovery_timing: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch, count: int,
) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    publications = []
    total_bytes = 0
    for index in range(count):
        payload = json.loads(protocol.worker.blob)
        payload["tensors"][0]["shards"][0]["addr"] += index * 4096
        blob = json.dumps(payload).encode()
        digest = hashlib.sha256(blob).hexdigest()
        protocol.worker.blobs[digest] = blob
        publication = refit_pb2.WeightVersionShard()
        publication.CopyFrom(protocol.control.shards[0])
        publication.worker_id = f"trainer-{index}"
        publication.logical_shard_id = f"rank:{index}"
        publication.stable_metadata_digest = digest
        publications.append(publication)
        total_bytes += len(blob)
    protocol.control.shards = publications
    with use_refit_timing(discovery_timing.recorder), telemetry.span("mx.refit.streaming_prepare") as parent:
        candidates = protocol.resolver.candidates(protocol.version)
        resolved = next(candidates)
        assert len(resolved.snapshot.shards) == count
        assert trace.get_current_span() is parent
        resolution, = _resolution_spans(recording)
        assert resolution.attributes["manifest_fetch_count"] == count
        assert resolution.attributes["manifest_bytes"] == total_bytes
        candidates.close()
    _assert_exported_counters(exported_metrics, discovery_timing.recorder.as_dict()["stages"]["source_preparation"])
    assert len(protocol.worker.stable_requests) == count
    assert protocol.worker.version_requests == []
    assert protocol.control.list_calls == 1


@pytest.mark.parametrize("harness", [StreamingSettings(1024, "cpu", 2)], indirect=True)
def test_prefetched_wire_timing_excludes_consumer_pause_with_current_values(
    recording, harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import modelexpress_rl.inference.nixl_staged_transfer as transfer_module

    prepared = harness.prepare()
    now = [0.0]
    monkeypatch.setattr(transfer_module.time, "perf_counter", lambda: now[0])
    original_post = prepared.transport.post_reads
    original_wait = prepared.transport.await_reads

    def post_reads(descriptors) -> list:
        posted = original_post(descriptors)
        now[0] += 2
        return posted

    def await_reads(posted) -> None:
        original_wait(posted)
        now[0] += 3

    monkeypatch.setattr(prepared.transport, "post_reads", post_reads)
    monkeypatch.setattr(prepared.transport, "await_reads", await_reads)
    metrics, installed = {}, {}
    with telemetry.span("mx.refit.receive"):
        for index, tensors in enumerate(harness.transfer.iter_bounded(prepared, metrics)):
            installed.update({name: tensor.clone() for name, tensor in tensors.items()})
            if index + 1 < metrics["batches"]:
                now[0] += 100
    test_refit_bounded_descriptors._check_values(harness, installed)
    assert metrics["staging_buffers"] == 2
    assert metrics["batches"] > 1
    assert metrics["wire_host_s"] == 5 * metrics["batches"]
    assert metrics["wire_wait_s"] == 3 * metrics["batches"]
    assert now[0] == 5 * metrics["batches"] + 100 * (metrics["batches"] - 1)


def test_real_checksum_verification_exports_success_and_error_timing(
    recording, harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import MappingProxyType

    from modelexpress.refit.timing import RefitTimingRecorder

    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")
    valid = harness.prepare()
    valid_timing = RefitTimingRecorder(backend="rl_generator", version="valid", rank=0)
    with use_refit_timing(valid_timing), telemetry.span("mx.refit.receive"):
        _, installed = harness.collect(valid)
    test_refit_bounded_descriptors._check_values(harness, installed)
    valid_stage = valid_timing.as_dict()["stages"]["receive_sync"]
    verification = [span for span in recording.get_finished_spans()
                    if span.name == "mx.refit.digest_verification"]
    assert len(verification) == len(valid.batches)
    assert all(span.attributes["status"] == "ok" for span in verification)
    assert valid_stage["status"] == "ok"
    assert valid_stage["metadata"]["digest_verification_s"] >= 0

    snapshot = harness.transfer.cached_trainer_source()
    corrupted = harness.transfer.prepare_streaming(
        manifests=[shard.metadata for shard in snapshot.shards],
        trainer_snapshot=snapshot,
        capture_layout=lambda _: (harness.capture, harness.layout),
        checksums=MappingProxyType({key: "incorrect" for key in valid.checksums}),
    )
    failed_timing = RefitTimingRecorder(backend="rl_generator", version="corrupted", rank=0)
    recording.clear()
    with use_refit_timing(failed_timing), telemetry.span("mx.refit.receive") as parent:
        with pytest.raises(RuntimeError, match="digest mismatch"):
            harness.collect(corrupted)
        assert trace.get_current_span() is parent
    failed_stage = failed_timing.as_dict()["stages"]["receive_sync"]
    verification, = [span for span in recording.get_finished_spans()
                     if span.name == "mx.refit.digest_verification"]
    assert verification.attributes["status"] == "error"
    assert verification.status.status_code is trace.StatusCode.ERROR
    assert failed_stage["status"] == "error"
    assert failed_stage["metadata"]["digest_verification_s"] >= 0


@pytest.mark.parametrize("pack", [False, True])
def test_multi_owner_bounded_compile_exports_one_aggregate_span_with_current_values(
    recording, harness, pack: bool,
) -> None:
    with telemetry.span("mx.refit.streaming_prepare") as parent:
        prepared = harness.prepare()
    metrics, installed = harness.collect(prepared)
    test_refit_bounded_descriptors._check_values(harness, installed)
    compilation, = [span for span in recording.get_finished_spans()
                    if span.name in {
                        "mx.refit.bounded_plan_compile",
                        "mx.refit.transfer_planning",
                        "mx.refit.transfer_validation",
                    }]
    assert compilation.name == "mx.refit.bounded_plan_compile"
    assert compilation.parent.span_id == parent.get_span_context().span_id
    assert compilation.attributes["module_batches"] == len(harness.layout) == 4
    assert compilation.attributes["batches"] == metrics["batches"]
    assert compilation.attributes["owner_plan_builds"] == 4
    assert compilation.attributes["bounded_whole_plan_builds"] == 0
    assert compilation.attributes["initial_whole_plan_s"] >= 0
    assert compilation.attributes["plan_cache_hits"] == 0
    assert compilation.attributes["plan_cache_misses"] == 1
    if pack:
        assert metrics["batches"] < 4
    else:
        assert metrics["batches"] == 4

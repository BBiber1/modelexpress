# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

pytest.importorskip("opentelemetry.sdk")
from modelexpress import telemetry
from modelexpress.refit.timing import RefitTimingRecorder, use_refit_timing
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF


@pytest.fixture
def recording(monkeypatch) -> Iterator[InMemorySpanExporter]:
    monkeypatch.setenv("MX_REFIT_TRACE_DETAIL", "1")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://unused/v1/traces")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", raising=False)
    monkeypatch.setattr(telemetry, "_configured_pid", os.getpid())
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("nixl-test"))
    yield exporter
    provider.shutdown()


def sample(size=64, duration=2000000) -> SimpleNamespace:
    return SimpleNamespace(
        totalBytes=size,
        descCount=2,
        startTime=1000,
        postDuration=100000,
        xferDuration=duration,
    )


@pytest.mark.parametrize("detail", [False, True])
@pytest.mark.parametrize("peers", [("trainer-0", "trainer-0"), ("trainer-0", "trainer-1")])
@pytest.mark.parametrize("logical_ids", [
    (None, None), ("slot-0", "slot-0"), ("slot-0", "slot-1"), ("slot-0", None)
])
def test_batch_encloses_distinct_completions_and_aggregates(
    recording, monkeypatch, detail, peers, logical_ids
) -> None:
    monkeypatch.setenv("MX_REFIT_TRACE_DETAIL", str(int(detail)))
    agent = SimpleNamespace(
        get_xfer_telemetry=lambda handle: sample(handle, handle * 10000)
    )
    with (
        telemetry.refit_attributes(
            {"role": "generator", "rank": 3, "step": 7, "experiment": "device"}
        ),
        telemetry.span("mx.refit") as refit,
    ):
        batch = telemetry._NixlBatch()
        with batch.posting():
            posted = []
            for peer, logical_id in zip(peers, logical_ids, strict=True):
                with telemetry.refit_attributes(
                    {"nixl.logical_shard_id": logical_id} if logical_id is not None else {}
                ):
                    posted.append(batch.post(peer))
            first, second = posted
            assert trace.get_current_span() is refit
        batch.complete(first, agent, 64)
        finished = recording.get_finished_spans()
        assert not finished
        batch.complete(second, agent, 128)
        batch._finish()
        timing = RefitTimingRecorder(backend="rl_generator", version="v7", rank=0)
        with use_refit_timing(timing), timing.span("control_discovery"):
            pass
    spans = recording.get_finished_spans()
    phase = next(s for s in spans if s.name == "mx.refit.control_discovery")
    assert phase.attributes["rank"] == 3
    assert phase.attributes["step"] == 7
    assert phase.attributes["version_uid"] == "v7"
    parent = next(s for s in spans if s.name == "mx.refit.nixl_batch")
    children = [s for s in spans if s.name == "mx.refit.nixl_transfer"]
    assert parent.parent.span_id == refit.get_span_context().span_id
    assert all(s.parent.span_id == parent.context.span_id for s in children)
    assert len(children) == (2 if detail else 0)
    if peers[0] == peers[1]:
        assert parent.attributes["nixl.remote_agent"] == peers[0]
    else:
        assert "nixl.remote_agent" not in parent.attributes
    assert "nixl.destination_agent" not in parent.attributes
    if logical_ids[0] is not None and logical_ids[0] == logical_ids[1]:
        assert parent.attributes["nixl.logical_shard_id"] == logical_ids[0]
    else:
        assert "nixl.logical_shard_id" not in parent.attributes
    if detail:
        assert tuple(s.attributes["nixl.remote_agent"] for s in children) == peers
        assert tuple(s.attributes.get("nixl.logical_shard_id") for s in children) == logical_ids
        assert all("nixl.destination_agent" not in s.attributes for s in children)
        assert (
            parent.start_time
            <= children[0].start_time
            <= children[0].end_time
            <= children[1].end_time
        )
        assert parent.end_time == children[1].end_time
    assert all(
        s.attributes["rank"] == 3 and s.attributes["step"] == 7
        for s in [parent, *children]
    )
    assert parent.attributes["nixl.total_bytes"] == 192
    assert parent.attributes["nixl.desc_count"] == 4
    assert parent.attributes["nixl.post_duration_s"] == pytest.approx(0.2)
    assert parent.attributes["nixl.xfer_duration_median_s"] == pytest.approx(0.96)
    assert parent.attributes["nixl.xfer_duration_p95_s"] == pytest.approx(1.248)
    assert parent.attributes["nixl.telemetry.complete"]
    assert parent.attributes["nixl.observed_batch_duration_s"] > 0
    assert parent.attributes["nixl.observed_payload_gbps"] > 0


@pytest.mark.parametrize("failure", ["unavailable", "transfer"])
def test_incomplete_native_coverage_and_failure(recording, failure) -> None:
    batch = telemetry._NixlBatch()
    with batch.posting():
        first, second = batch.post("one"), batch.post("two")
    batch.complete(
        first, SimpleNamespace(get_xfer_telemetry=lambda h: sample()), object()
    )
    if failure == "transfer":
        batch.fail(second, RuntimeError("failed"))
    else:

        def unavailable(handle) -> None:
            raise RuntimeError("no telemetry")

        batch.complete(
            second, SimpleNamespace(get_xfer_telemetry=unavailable), object()
        )
    batch._finish()
    parent = next(
        s for s in recording.get_finished_spans() if s.name == "mx.refit.nixl_batch"
    )
    assert parent.attributes["nixl.request_count"] == 2
    assert parent.attributes["nixl.telemetry_count"] == 1
    assert parent.attributes["nixl.completed_count"] == (
        1 if failure == "transfer" else 2
    )
    assert parent.attributes["nixl.total_bytes"] == 64
    assert not parent.attributes["nixl.telemetry.complete"]
    assert "nixl.observed_payload_gbps" not in parent.attributes
    assert (parent.status.status_code == trace.StatusCode.ERROR) == (
        failure == "transfer"
    )
    requests = [s for s in recording.get_finished_spans() if s.name == "mx.refit.nixl_transfer"]
    assert [s.attributes["nixl.remote_agent"] for s in requests] == ["one", "two"]


def test_mixed_shard_batch_does_not_inherit_a_single_logical_id(recording) -> None:
    with telemetry.refit_attributes({"nixl.logical_shard_id": "inherited"}):
        batch = telemetry._NixlBatch()
    agent = SimpleNamespace(get_xfer_telemetry=lambda h: sample())
    with batch.posting():
        for slot in ("left", "right"):
            with telemetry.refit_attributes({"nixl.logical_shard_id": slot}):
                request = batch.post(slot + "-agent")
            batch.complete(request, agent, object())
    spans = recording.get_finished_spans()
    parent = next(s for s in spans if s.name == "mx.refit.nixl_batch")
    assert "nixl.logical_shard_id" not in parent.attributes
    assert [s.attributes["nixl.logical_shard_id"] for s in spans
            if s.name == "mx.refit.nixl_transfer"] == ["left", "right"]


def test_serial_batch_remains_open_between_requests(recording) -> None:
    batch = telemetry._NixlBatch()
    agent = SimpleNamespace(get_xfer_telemetry=lambda h: sample())
    with batch.posting():
        first = batch.post("one")
        batch.complete(first, agent, object())
        assert all(
            s.name != "mx.refit.nixl_batch" for s in recording.get_finished_spans()
        )
        second = batch.post("two")
        batch.complete(second, agent, object())
    assert (
        len(
            [
                s
                for s in recording.get_finished_spans()
                if s.name == "mx.refit.nixl_batch"
            ]
        )
        == 1
    )


def test_disabled_or_nonrecording_skips_request_work(recording, monkeypatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    with telemetry._NixlBatch().posting():
        assert telemetry._NixlBatch().post("peer") is None
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://unused")
    provider = TracerProvider(sampler=ALWAYS_OFF)
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("off"))
    batch = telemetry._NixlBatch()
    assert batch.post("peer") is None
    assert not recording.get_finished_spans()
    cycle = telemetry.RefitCycle()
    carrier = {}
    cycle.inject(carrier)
    assert not int(carrier["traceparent"].rsplit("-", 1)[1], 16) & 1
    with cycle.active():
        with telemetry.span("mx.refit.publish"):
            assert not telemetry.recording()
    cycle.finish()
    assert not recording.get_finished_spans()
    provider.shutdown()


def test_manager_collects_each_completion_before_any_release(recording) -> None:
    pytest.importorskip("torch")
    from modelexpress.nixl_transfer import NixlTransferManager, PostedRead

    manager = object.__new__(NixlTransferManager)
    manager._transfer_traces = {}
    events = []
    handles = [object(), object()]
    polled = {id(h): 0 for h in handles}

    def state(handle) -> str:
        polled[id(handle)] += 1
        if handle is handles[1] and polled[id(handle)] == 1:
            assert events == [("telemetry", handles[0])]
            return "IN_PROGRESS"
        return "DONE"

    def native(handle) -> SimpleNamespace:
        assert not any(e[0] == "release" for e in events)
        events.append(("telemetry", handle))
        return sample()

    manager._agent = SimpleNamespace(
        check_xfer_state=state,
        get_xfer_telemetry=native,
        release_xfer_handle=lambda h: events.append(("release", h)),
    )
    manager._accelerator_backend = SimpleNamespace(
        synchronize=lambda d: events.append(("sync", d))
    )
    manager._device_id = 0
    with telemetry._NixlBatch().posting():
        for h in handles:
            with telemetry.refit_attributes({"nixl.logical_shard_id": "slot"}):
                manager._trace_post(h, "trainer")
    posted = [PostedRead(h, "trainer", 64, 1) for h in handles]
    assert manager.await_read_batches(posted)[:2] == (128, 2)
    requests = [
        s for s in recording.get_finished_spans()
        if s.name == "mx.refit.nixl_transfer"
    ]
    assert len(requests) == 2
    assert all(s.attributes["nixl.remote_agent"] == "trainer" for s in requests)
    assert all(s.attributes["nixl.logical_shard_id"] == "slot" for s in requests)
    assert [e[0] for e in events] == [
        "telemetry",
        "telemetry",
        "release",
        "release",
        "sync",
    ]


def test_native_metric_units_and_request_span_exemplars(recording, monkeypatch) -> None:
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    telemetry._value_histogram.cache_clear()
    telemetry._histogram.cache_clear()
    monkeypatch.setattr(metrics, "get_meter", provider.get_meter)
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://unused/v1/metrics"
    )
    try:
        with telemetry.refit_attributes({"role": "generator", "rank": 7}):
            batch = telemetry._NixlBatch()
            request = batch.post("trainer")
        # A prefetched read can complete after its posting context has been detached.
        batch.complete(
            request, SimpleNamespace(get_xfer_telemetry=lambda h: sample()), object()
        )
        batch._finish()
        exported = [
            m
            for r in reader.get_metrics_data().resource_metrics
            for scope in r.scope_metrics
            for m in scope.metrics
        ]
        values = {m.name: m for m in exported}
        child = next(
            s
            for s in recording.get_finished_spans()
            if s.name == "mx.refit.nixl_transfer"
        )
        parent = next(
            s for s in recording.get_finished_spans() if s.name == "mx.refit.nixl_batch"
        )
        for name, span, unit in (
            ("mx_refit_nixl_request_total_bytes", child, "By"),
            ("mx_refit_nixl_request_xfer_duration", child, "s"),
            ("mx_refit_nixl_batch_total_bytes", parent, "By"),
            ("mx_refit_nixl_observed_batch_duration", parent, "s"),
        ):
            metric = values[name]
            assert metric.unit == unit
            point = metric.data.data_points[0]
            assert point.attributes["rank"] == 7
            assert point.attributes["role"] == "generator"
            assert any(
                e.trace_id == span.context.trace_id
                and e.span_id == span.context.span_id
                for e in point.exemplars
            )
        assert (
            values["mx_refit_nixl_request_xfer_duration"].data.data_points[0].sum == 2
        )
    finally:
        provider.shutdown()
        telemetry._value_histogram.cache_clear()
        telemetry._histogram.cache_clear()


def test_batch_parent_is_captured_before_wire_post(recording) -> None:
    agent = SimpleNamespace(get_xfer_telemetry=lambda _handle: sample())
    with telemetry.span("mx.refit.stage") as stage:
        batch = telemetry._NixlBatch()
        with batch.posting():
            with telemetry.span("mx.refit.wire_post") as post:
                request = batch.post("trainer")
        batch.complete(request, agent, object())
        batch._finish()

    spans = recording.get_finished_spans()
    parent = next(s for s in spans if s.name == "mx.refit.nixl_batch")
    post = next(s for s in spans if s.name == "mx.refit.wire_post")
    assert parent.parent.span_id == stage.get_span_context().span_id
    assert parent.parent.span_id != post.context.span_id
    assert parent.end_time >= post.end_time


def test_completed_wait_preserves_timestamp_parent_and_worker_identity(
    recording,
) -> None:
    with (
        telemetry.refit_attributes({"role": "generator", "rank": 7}),
        telemetry.span("mx.refit", start_time=1_000_000) as refit,
        telemetry.refit_attributes({"role": "generator", "rank": 0, "step": 2}),
    ):
        telemetry.completed_span("mx.refit.wait_version_marker", 2_000_000, 3_000_000)
        with telemetry.span("mx.refit.nested_phase"):
            pass
    spans = recording.get_finished_spans()
    wait = next(s for s in spans if s.name == "mx.refit.wait_version_marker")
    assert (wait.start_time, wait.end_time) == (2_000_000, 3_000_000)
    assert wait.parent.span_id == refit.get_span_context().span_id
    assert all(s.attributes["rank"] == 7 for s in spans)
    assert next(s for s in spans if s.name == "mx.refit").start_time == 1_000_000


@pytest.mark.parametrize("configured", [False, True])
def test_standalone_native_batch_exports_without_refit_scope(
    monkeypatch, configured
) -> None:
    from opentelemetry.exporter.otlp.proto.http import trace_exporter

    monkeypatch.setenv("MX_REFIT_TRACE_DETAIL", "1")
    exporter = InMemorySpanExporter()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://unused/v1/traces")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", raising=False)
    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", lambda **_kwargs: exporter)
    monkeypatch.setattr(telemetry, "_configured_pid", None)
    monkeypatch.setattr(telemetry, "_tracer", None)
    monkeypatch.setattr(telemetry, "_tracer_provider", None)
    if configured:
        telemetry.configure("existing-role")
    batch = telemetry._NixlBatch()
    try:
        with batch.posting():
            request = batch.post("trainer")
        batch.complete(request, SimpleNamespace(get_xfer_telemetry=lambda _handle: sample()), 1)
        batch._finish()
        assert telemetry._tracer_provider.force_flush(timeout_millis=5000)
        spans = exporter.get_finished_spans()
        assert {span.name for span in spans} == {
            "mx.refit.nixl_batch", "mx.refit.nixl_transfer"
        }
        assert all(
            span.resource.attributes["service.name"]
            == ("existing-role" if configured else "modelexpress")
            for span in spans
        )
    finally:
        telemetry._tracer_provider.shutdown()


def test_refit_labels_use_registered_environment_defaults(monkeypatch) -> None:
    monkeypatch.setenv("MX_REFIT_EXPERIMENT", "run-a")
    monkeypatch.setenv("MX_REFIT_STAGING_MODE", "bounded")
    metadata = telemetry.refit_metadata(step=2)
    assert metadata["experiment"] == "run-a"
    assert metadata["staging_mode"] == "bounded"
    assert telemetry.refit_metadata(staging_mode="full")["staging_mode"] == "full"

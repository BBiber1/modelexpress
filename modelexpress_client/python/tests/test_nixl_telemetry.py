# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("opentelemetry.sdk")
from modelexpress import telemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF


@pytest.fixture
def recording(monkeypatch):
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


def sample(size=64, duration=2000000):
    return SimpleNamespace(
        totalBytes=size,
        descCount=2,
        startTime=1000,
        postDuration=100000,
        xferDuration=duration,
    )


@pytest.mark.parametrize("detail", [False, True])
def test_batch_encloses_distinct_completions_and_aggregates(
    recording, monkeypatch, detail
):
    monkeypatch.setenv("MX_REFIT_TRACE_DETAIL", str(int(detail)))
    events = []
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
            first, second = batch.post("trainer-0"), batch.post("trainer-1")
            assert trace.get_current_span() is refit
        batch.complete(first, agent, 64)
        finished = recording.get_finished_spans()
        assert not finished
        events.append(batch._requests[first]["end_ns"])
        batch.complete(second, agent, 128)
        batch._finish()
    spans = recording.get_finished_spans()
    parent = next(s for s in spans if s.name == "mx.refit.nixl_batch")
    children = [s for s in spans if s.name == "mx.refit.nixl_transfer"]
    assert parent.parent.span_id == refit.get_span_context().span_id
    assert all(s.parent.span_id == parent.context.span_id for s in children)
    assert len(children) == (2 if detail else 0)
    if detail:
        assert (
            parent.start_time
            <= children[0].start_time
            <= events[0]
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
    assert parent.attributes["nixl.payload_gbps"] > 0


@pytest.mark.parametrize("failure", ["unavailable", "transfer"])
def test_incomplete_native_coverage_and_failure(recording, failure):
    batch = telemetry._NixlBatch()
    with batch.posting():
        first, second = batch.post("one"), batch.post("two")
    batch.complete(
        first, SimpleNamespace(get_xfer_telemetry=lambda h: sample()), object()
    )
    if failure == "transfer":
        batch.fail(second, RuntimeError("failed"))
    else:

        def unavailable(handle):
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
    assert "nixl.payload_gbps" not in parent.attributes
    assert (parent.status.status_code == trace.StatusCode.ERROR) == (
        failure == "transfer"
    )


def test_serial_batch_remains_open_between_requests(recording):
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


def test_disabled_or_nonrecording_skips_request_work(recording, monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    with telemetry._NixlBatch().posting():
        assert telemetry._NixlBatch().post("peer") is None
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://unused")
    provider = TracerProvider(sampler=ALWAYS_OFF)
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("off"))
    batch = telemetry._NixlBatch()
    assert batch.post("peer") is None
    assert not batch._requests
    assert not recording.get_finished_spans()
    cycle = telemetry.RefitCycle()
    carrier = {}
    cycle.inject(carrier)
    assert carrier["traceparent"].endswith("-00")
    with cycle.active():
        with telemetry.span("mx.refit.publish"):
            assert not telemetry.recording()
    cycle.finish()
    assert not recording.get_finished_spans()
    provider.shutdown()


def test_manager_collects_each_completion_before_any_release(recording):
    pytest.importorskip("torch")
    from modelexpress.nixl_transfer import NixlTransferManager, PostedRead

    manager = object.__new__(NixlTransferManager)
    events = []
    handles = [object(), object()]
    polled = {id(h): 0 for h in handles}

    def state(handle):
        polled[id(handle)] += 1
        if handle is handles[1] and polled[id(handle)] == 1:
            assert events == [("telemetry", handles[0])]
            return "IN_PROGRESS"
        return "DONE"

    def native(handle):
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
            manager._trace_post(h, "trainer")
    posted = [PostedRead(h, "trainer", 64, 1) for h in handles]
    assert manager.await_read_batches(posted)[:2] == (128, 2)
    assert [e[0] for e in events] == [
        "telemetry",
        "telemetry",
        "release",
        "release",
        "sync",
    ]
    assert not manager._transfer_traces


def test_native_metric_units_and_request_span_exemplars(recording, monkeypatch):
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
            ("mx_refit_nixl_batch_duration", parent, "s"),
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


def test_completed_wait_preserves_timestamp_parent_and_worker_identity(recording):
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

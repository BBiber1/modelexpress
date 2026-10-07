import asyncio
import importlib.util
import multiprocessing
import os
import random
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http import trace_exporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.id_generator import IdGenerator


class ForkStateIdGenerator(IdGenerator):
    def __init__(self):
        self.next_id = 1

    def generate_span_id(self):
        self.next_id += 1
        return self.next_id

    def generate_trace_id(self):
        return 123


def child(module, carrier, queue):
    module.configure("mx-fork-test")
    random.seed(1234)  # vLLM seeds every worker identically after fork.
    with (
        module.extracted(carrier),
        module.span("mx.refit", {"role": "generator", "rank": os.getpid()}) as span,
    ):
        context = span.get_span_context()
        queue.put(
            (
                context.trace_id,
                context.span_id,
                span.is_recording(),
                span.parent.span_id,
            )
        )


def main():
    path = Path(__file__).resolve().parents[1] / "modelexpress/telemetry.py"
    spec = importlib.util.spec_from_file_location("mx_telemetry_fork_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://127.0.0.1:9/v1/traces"
    trace.set_tracer_provider(TracerProvider(id_generator=ForkStateIdGenerator()))
    with patch.object(
        trace_exporter, "OTLPSpanExporter", lambda endpoint: InMemorySpanExporter()
    ):
        module.configure("mx-fork-test")
        cycle = module.RefitCycle({"role": "trainer", "rank": 0, "step": 3})
        carrier = {}
        cycle.inject(carrier)
        _, trace_id, span_id, _ = carrier["traceparent"].split("-")
        parent_id, cycle_span_id = int(trace_id, 16), int(span_id, 16)
        with module.extracted(carrier), module.span("mx.refit.offer") as offer:
            assert offer.parent.span_id == cycle_span_id
        assert cycle.is_recording()
        context = multiprocessing.get_context("fork")
        queue = context.Queue()
        children = [
            context.Process(target=child, args=(module, carrier, queue))
            for _ in range(8)
        ]
        for process in children:
            process.start()
        values = [queue.get(timeout=10) for _ in children]
        for process in children:
            process.join(timeout=10)
            assert process.exitcode == 0
        assert {value[0] for value in values} == {parent_id}
        assert len({value[1] for value in values}) == 8
        assert all(value[2] for value in values)
        assert {value[3] for value in values} == {cycle_span_id}
        assert cycle.is_recording()
        cycle.finish()
        assert not cycle.is_recording()
        print("eight forked ranks have distinct, correlated, recording span IDs")


def test_forked_span_ids():
    subprocess.run([sys.executable, __file__, "check"], check=True, timeout=30)


@pytest.mark.parametrize("failed", [False, True])
def test_native_cycle_lifetime_and_error(failed, monkeypatch):
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    path = Path(__file__).resolve().parents[1] / "modelexpress/telemetry.py"
    spec = importlib.util.spec_from_file_location("mx_cycle_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(module.RefitSpanProcessor(SimpleSpanProcessor(exporter)))
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://unused/v1/traces")
    module._configured_pid = os.getpid()
    module._tracer = provider.get_tracer("cycle-test")
    with module.span("ambient") as ambient:
        cycle = module.RefitCycle({"role": "trainer", "rank": 0, "step": 3})
        carrier = {}
        cycle.inject(carrier)
        assert trace.get_current_span() is ambient
        with module.extracted(carrier):
            with module.span("mx.refit.offer"):
                pass
            for role in ("trainer", "control", "generator", "server"):
                with module.span("mx.refit", {"role": role, "rank": 0}):
                    pass
        assert cycle.is_recording()
        cycle.finish(RuntimeError("refit failed") if failed else None)
        cycle.finish()  # Completion is idempotent.
    recorded = exporter.get_finished_spans()
    root = next(s for s in recorded if s.name == "mx.refit.cycle")
    assert root.parent is None
    assert root.context.trace_id != ambient.get_span_context().trace_id
    children = [s for s in recorded if s.name not in ("mx.refit.cycle", "ambient")]
    assert len(children) == 5
    assert all(s.parent.span_id == root.context.span_id for s in children)
    assert all(
        root.start_time <= s.start_time <= s.end_time <= root.end_time for s in children
    )
    assert root.attributes["status"] == ("failed" if failed else "complete")
    assert (root.status.status_code == trace.StatusCode.ERROR) == failed
    assert bool(root.events) == failed
    provider.shutdown()


def test_disabled_cycle_skips_context_work(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "modelexpress/telemetry.py"
    spec = importlib.util.spec_from_file_location("mx_cycle_disabled_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    cycle = module.RefitCycle({})
    carrier = {}
    cycle.inject(carrier)
    assert not cycle.is_recording() and not carrier
    module.completed_span("disabled", 100, 200, error=asyncio.CancelledError())
    assert module._tracer is None
    cycle.finish()


if __name__ == "__main__":
    main()


def test_otlp_http_exports_correlated_success_and_failure(monkeypatch):
    import http.server
    import threading

    from modelexpress import telemetry
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    payloads = []

    class Collector(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            message = ExportTraceServiceRequest()
            message.ParseFromString(self.rfile.read(int(self.headers["Content-Length"])))
            payloads.append(message)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", f"http://127.0.0.1:{server.server_port}/v1/traces")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", raising=False)
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "mx.refit.run_id=export-test")
    monkeypatch.setattr(telemetry, "_configured_pid", None)
    monkeypatch.setattr(telemetry, "_tracer", None)
    monkeypatch.setattr(telemetry, "_tracer_provider", None)
    try:
        telemetry.configure("mx-export-test")
        cycle = telemetry.RefitCycle({"step": 0, "refit.root": True})
        carrier = {}
        cycle.inject(carrier)
        with telemetry.extracted(carrier), telemetry.span("mx.refit", {"role": "generator", "rank": 0}):
            pass
        error = RuntimeError("installation failed")
        with pytest.raises(RuntimeError), telemetry.extracted(carrier), telemetry.span("mx.refit.install"):
            raise error
        cycle.finish(error)
        assert telemetry._tracer_provider.force_flush(timeout_millis=5000)
        spans = [span for message in payloads for resource in message.resource_spans for scope in resource.scope_spans for span in scope.spans]
        root = next(span for span in spans if span.name == "mx.refit.cycle")
        children = [span for span in spans if span.name != "mx.refit.cycle"]
        assert len(children) == 2
        assert all(span.trace_id == root.trace_id and span.parent_span_id == root.span_id for span in children)
        assert root.status.code == 2
        assert next(span for span in spans if span.name == "mx.refit.install").status.code == 2
    finally:
        if telemetry._tracer_provider is not None:
            telemetry._tracer_provider.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("failure", [None, RuntimeError("preparation failed"), asyncio.CancelledError()])
def test_role_envelopes_and_shared_attributes(monkeypatch, failure):
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    path = Path(__file__).resolve().parents[1] / "modelexpress/telemetry.py"
    spec = importlib.util.spec_from_file_location("mx_envelope_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(module.RefitSpanProcessor(SimpleSpanProcessor(exporter)))
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://unused/v1/traces")
    module._configured_pid = os.getpid()
    module._tracer = provider.get_tracer("envelope-test")
    root = module.RefitCycle({"step": 3, "experiment": "test", "version_uid": "v3"})
    carrier = {}
    root.inject(carrier)
    with module.extracted(carrier), module.refit_attributes(role="trainer", rank=7):
        with module.refit_span("mx.refit.trainers") as group:
            with module.refit_span("mx.refit.trainer") as trainer:
                module.completed_span("publish", 100, 200, error=failure)
                module.completed_span("release", 180, 300)
            group.include([50, 400])
    root.finish()
    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert trainer.interval == [100, 300]
    assert group.interval == root.interval == [50, 400]
    assert (spans["mx.refit.trainer"].start_time, spans["mx.refit.trainer"].end_time) == (100, 300)
    assert (spans["mx.refit.cycle"].start_time, spans["mx.refit.cycle"].end_time) == (50, 400)
    assert spans["publish"].attributes["rank"] == 7
    assert spans["publish"].attributes["version_uid"] == "v3"
    if failure is not None:
        assert spans["publish"].status.status_code == trace.StatusCode.ERROR
        assert spans["publish"].events[0].name == "exception"
        assert spans["publish"].attributes["status"] == (
            "cancelled" if isinstance(failure, asyncio.CancelledError) else "failed"
        )
    else:
        assert spans["publish"].status.status_code == trace.StatusCode.UNSET
        assert not spans["publish"].events
    with module.extracted(carrier), module.refit_attributes(role="generator", rank=5):
        with module.refit_attributes({"role": "trainer", "rank": 0}), module.span("nested") as nested:
            assert nested.attributes["role"] == "generator" and nested.attributes["rank"] == 5
    assert not module._refit_attributes.get()
    with module.span("ambient") as ambient:
        with pytest.raises(ValueError), module.extracted({"baggage": "step=bad"}):
            pass
        assert trace.get_current_span() is ambient
    provider.shutdown()

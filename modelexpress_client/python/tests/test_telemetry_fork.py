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
        queue.put((context.trace_id, context.span_id, span.is_recording()))


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
        with module.span("mx.refit.parent") as parent:
            carrier = {}
            module.inject(carrier)
            parent_id = parent.get_span_context().trace_id
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
        print("eight forked ranks have distinct, correlated, recording span IDs")


def test_forked_span_ids():
    subprocess.run([sys.executable, __file__, "check"], check=True, timeout=30)


if __name__ == "__main__":
    main()

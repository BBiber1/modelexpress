# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional OpenTelemetry support for ModelExpress refits.

Callers use this module without importing OpenTelemetry directly. Installing the
``otel`` extra and setting OTLP endpoints enables export; otherwise it is inert.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import secrets
import time
from collections.abc import Iterator, Mapping, MutableMapping
from functools import lru_cache
from typing import Any

_configured_pid: int | None = None
_tracer: Any | None = None
_tracer_provider: Any | None = None
_refit_attributes: contextvars.ContextVar[Mapping[str, str | int]] = (
    contextvars.ContextVar("mx_refit_attributes", default={})
)


def configure(service_name: str) -> None:
    """Configure this process from standard OTLP HTTP environment variables."""
    global _configured_pid, _tracer, _tracer_provider
    if _configured_pid == os.getpid():
        return
    traces = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    metrics = os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
    if not traces and not metrics:
        return
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.sdk.resources import Resource

    resource = Resource.create({"service.name": service_name})
    if traces:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
        from opentelemetry.sdk.trace.sampling import ALWAYS_ON

        class _ProcessIndependentIdGenerator(RandomIdGenerator):
            def generate_span_id(self) -> int:
                return secrets.randbits(64) or self.generate_span_id()

            def generate_trace_id(self) -> int:
                return secrets.randbits(128) or self.generate_trace_id()

        # vLLM seeds global random identically in its forked workers.
        provider = TracerProvider(
            resource=resource,
            sampler=ALWAYS_ON,
            id_generator=_ProcessIndependentIdGenerator(),
        )
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=traces))
        )
        _tracer_provider = provider
        _tracer = provider.get_tracer("modelexpress.refit")
    if metrics:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        otel_metrics.set_meter_provider(
            MeterProvider(
                metric_readers=[
                    PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=metrics))
                ],
                resource=resource,
            )
        )
    _configured_pid = os.getpid()


def enabled() -> bool:
    return bool(os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"))


def _process_tracer() -> Any:
    from opentelemetry import trace

    return (
        _tracer
        if _configured_pid == os.getpid() and _tracer is not None
        else trace.get_tracer("modelexpress.refit")
    )


class RefitCycle:
    """A native refit root whose lifetime spans offer and completion hooks."""

    def __init__(self, attributes: Mapping[str, Any]) -> None:
        self._span: Any | None = None
        if not enabled():
            return
        from opentelemetry import context

        self._span = _process_tracer().start_span(
            "mx.refit.cycle", context=context.Context()
        )
        if self.is_recording():
            self._span.set_attributes(dict(attributes))

    def is_recording(self) -> bool:
        return self._span is not None and self._span.is_recording()

    def inject(self, carrier: MutableMapping[str, str]) -> None:
        """Propagate the cycle itself, keeping offers and role spans siblings."""
        if not self.is_recording():
            return
        from opentelemetry import trace
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )

        TraceContextTextMapPropagator().inject(
            carrier, context=trace.set_span_in_context(self._span)
        )

    def finish(self, error: BaseException | None = None) -> None:
        if self._span is None:
            return
        if self.is_recording():
            self._span.set_attribute(
                "status", "failed" if error is not None else "complete"
            )
            if error is not None:
                from opentelemetry.trace import StatusCode

                self._span.record_exception(error)
                self._span.set_status(StatusCode.ERROR, str(error))
        self._span.end()
        self._span = None


@contextlib.contextmanager
def span(name: str, attributes: Mapping[str, Any] | None = None) -> Iterator[Any]:
    if not enabled():
        yield _NoopSpan()
        return
    tracer = _process_tracer()
    with tracer.start_as_current_span(name) as current:
        if current.is_recording():
            current.set_attributes(dict(_refit_attributes.get()))
            if attributes:
                current.set_attributes(dict(attributes))
        yield current


def inject(carrier: MutableMapping[str, str]) -> None:
    """Write W3C traceparent and tracestate to an HTTP or gRPC carrier."""
    if not enabled():
        return
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    TraceContextTextMapPropagator().inject(carrier)


@contextlib.contextmanager
def extracted(carrier: Mapping[str, str]) -> Iterator[None]:
    """Make an incoming W3C parent current for the enclosed work."""
    if not enabled():
        yield
        return
    from opentelemetry import context
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    token = context.attach(TraceContextTextMapPropagator().extract(carrier))
    try:
        yield
    finally:
        context.detach(token)


@lru_cache(maxsize=16)
def _histogram(name: str) -> Any:
    from opentelemetry import metrics

    return metrics.get_meter("modelexpress.refit").create_histogram(name, unit="s")


def duration(name: str, seconds: float, attributes: Mapping[str, Any]) -> None:
    if os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"):
        _histogram(name).record(seconds, dict(attributes))


@lru_cache(maxsize=32)
def _value_histogram(name: str) -> Any:
    from opentelemetry import metrics

    unit = "By" if name.endswith("bytes") else "1"
    return metrics.get_meter("modelexpress.refit").create_histogram(
        f"mx_refit_{name}", unit=unit
    )


@contextlib.contextmanager
def refit_attributes(attributes: Mapping[str, str | int]) -> Iterator[None]:
    """Apply role and rank to nested refit spans in the current context."""
    token = _refit_attributes.set(attributes)
    try:
        yield
    finally:
        _refit_attributes.reset(token)


def attribute(name: str, value: float) -> None:
    """Record a numeric refit fact on its span and as an OTLP metric."""
    if os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"):
        dimensions = {
            key: value
            for key, value in _refit_attributes.get().items()
            if key in ("role", "rank", "experiment", "staging_mode")
        }
        _value_histogram(name).record(
            int(value) if isinstance(value, bool) else value, dimensions
        )
    if enabled():
        from opentelemetry import trace

        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute(f"mx.measurement.{name}", value)


def refit_channel(channel: Any) -> Any:
    """Attach current W3C context to refit service unary RPCs."""
    import grpc

    from modelexpress.auth import _ClientCallDetails

    class _TraceInterceptor(grpc.UnaryUnaryClientInterceptor):
        def intercept_unary_unary(self, continuation, details, request):
            carrier: dict[str, str] = {}
            inject(carrier)
            if not carrier:
                return continuation(details, request)
            metadata = list(details.metadata or ())
            metadata.extend(carrier.items())
            return continuation(
                _ClientCallDetails(
                    details.method,
                    details.timeout,
                    metadata,
                    details.credentials,
                    details.wait_for_ready,
                    details.compression,
                ),
                request,
            )

    return grpc.intercept_channel(channel, _TraceInterceptor())


class _NoopSpan:
    def is_recording(self) -> bool:
        return False

    def set_attribute(self, _name: str, _value: Any) -> None:
        pass


_nixl_batch: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "mx_nixl_batch", default=None
)


class _NixlBatch:
    """Private asynchronous batch lifetime; no context stays attached while waiting."""

    def __init__(self) -> None:
        self._span: Any | None = None
        self._requests: dict[Any, dict[str, Any]] = {}
        self._posting = False
        self._error: BaseException | None = None
        self._start_ns = 0
        self._attributes: dict[str, Any] = {}

    @contextlib.contextmanager
    def posting(self) -> Iterator[None]:
        if not enabled():
            yield
            return
        token = _nixl_batch.set(self)
        self._posting = True
        try:
            yield
        except BaseException as error:
            self._error = error
            raise
        finally:
            self._posting = False
            _nixl_batch.reset(token)
            self._finish()

    def post(self, peer: str) -> Any | None:
        if not enabled():
            return None
        from opentelemetry import trace

        if self._span is None:
            self._start_ns = time.time_ns()
            self._span = _process_tracer().start_span(
                "mx.refit.nixl_batch", start_time=self._start_ns
            )
            if self._span.is_recording():
                self._attributes = dict(_refit_attributes.get())
                self._span.set_attributes(self._attributes)
        if not self._span.is_recording():
            return None
        request = _process_tracer().start_span(
            "mx.refit.nixl_transfer",
            context=trace.set_span_in_context(self._span),
        )
        if not request.is_recording():
            request.end()
            return None
        request.set_attributes(
            {
                **_refit_attributes.get(),
                "nixl.operation": "READ",
                "nixl.remote_agent": peer,
            }
        )
        self._requests[request] = {"end_ns": None, "complete": False, "native": None}
        return request

    def complete(self, request: Any, agent: Any, handle: Any) -> None:
        if not request.is_recording():
            return
        from opentelemetry import trace

        end_ns = time.time_ns()
        native = None
        try:
            sample = agent.get_xfer_telemetry(handle)
            native = {
                "total_bytes": int(sample.totalBytes),
                "desc_count": int(sample.descCount),
                "start_time_us": int(sample.startTime),
                "post_duration_s": int(sample.postDuration) / 1e6,
                "xfer_duration_s": int(sample.xferDuration) / 1e6,
            }
            request.set_attributes({"nixl." + k: v for k, v in native.items()})
            with trace.use_span(request, end_on_exit=False):
                self._metrics(native, "request")
        except Exception as error:
            request.set_attribute("nixl.telemetry.error", str(error))
        request.set_attribute("nixl.telemetry.available", native is not None)
        request.set_attribute("nixl.complete", True)
        self._requests[request] = {"end_ns": end_ns, "complete": True, "native": native}
        request.end(end_time=end_ns)
        self._finish()

    def fail(self, request: Any, error: BaseException) -> None:
        self._error = error
        if self._requests[request]["end_ns"] is not None:
            return
        from opentelemetry.trace import StatusCode

        end_ns = time.time_ns()
        request.set_status(StatusCode.ERROR, str(error))
        request.record_exception(error)
        request.set_attribute("nixl.complete", False)
        self._requests[request]["end_ns"] = end_ns
        request.end(end_time=end_ns)
        self._finish()

    def _metrics(self, values: Mapping[str, Any], kind: str) -> None:
        if not os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"):
            return
        dimensions = {
            k: v
            for k, v in self._attributes.items()
            if k in ("role", "rank", "experiment", "staging_mode")
        }
        dimensions["stage"] = kind
        try:
            for name in ("total_bytes", "desc_count"):
                _value_histogram("nixl_" + kind + "_" + name).record(
                    values[name], dimensions
                )
            for name in ("post_duration_s", "xfer_duration_s", "batch_duration_s"):
                if name in values:
                    duration(
                        "mx_refit_nixl_"
                        + (
                            "batch_duration"
                            if name == "batch_duration_s"
                            else kind + "_" + name.removesuffix("_s")
                        ),
                        values[name],
                        dimensions,
                    )
        except Exception:
            logging.getLogger(__name__).warning(
                "NIXL metric recording failed", exc_info=True
            )

    def _finish(self) -> None:
        if (
            self._span is None
            or not self._span.is_recording()
            or self._posting
            or not self._requests
            or any(r["end_ns"] is None for r in self._requests.values())
        ):
            return
        from opentelemetry import trace
        from opentelemetry.trace import StatusCode

        samples = [
            r["native"] for r in self._requests.values() if r["native"] is not None
        ]
        end_ns = max(r["end_ns"] for r in self._requests.values())
        elapsed = (end_ns - self._start_ns) / 1e9
        values = {
            "total_bytes": sum(s["total_bytes"] for s in samples),
            "desc_count": sum(s["desc_count"] for s in samples),
            "post_duration_s": sum(s["post_duration_s"] for s in samples),
        }
        attrs = {"nixl." + k: v for k, v in values.items()}
        attrs.update(
            {
                "nixl.request_count": len(self._requests),
                "nixl.completed_count": sum(
                    r["complete"] for r in self._requests.values()
                ),
                "nixl.telemetry_count": len(samples),
                "nixl.telemetry.complete": len(samples) == len(self._requests),
                "nixl.batch_duration_s": elapsed,
            }
        )
        if samples:
            times = sorted(s["xfer_duration_s"] for s in samples)
            for name, fraction in (
                ("min", 0),
                ("median", 0.5),
                ("max", 1),
                ("p95", 0.95),
            ):
                position = (len(times) - 1) * fraction
                lo = int(position)
                hi = min(lo + 1, len(times) - 1)
                attrs["nixl.xfer_duration_" + name + "_s"] = times[lo] + (
                    times[hi] - times[lo]
                ) * (position - lo)
            if (
                elapsed > 0
                and len(samples) == len(self._requests)
                and self._error is None
            ):
                attrs["nixl.payload_gbps"] = 8 * values["total_bytes"] / elapsed / 1e9
        self._span.set_attributes(attrs)
        if self._error is not None:
            self._span.set_status(StatusCode.ERROR, str(self._error))
            self._span.record_exception(self._error)
        with trace.use_span(self._span, end_on_exit=False):
            self._metrics({**values, "batch_duration_s": elapsed}, "batch")
        self._span.end(end_time=end_ns)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional OpenTelemetry support for ModelExpress refits.

Callers use this module without importing OpenTelemetry directly. Installing the
``otel`` extra and setting OTLP endpoints enables export; otherwise it is inert.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
from collections.abc import Iterator, Mapping, MutableMapping
from functools import lru_cache
from typing import Any

_configured_pid: int | None = None
_refit_attributes: contextvars.ContextVar[Mapping[str, str | int]] = (
    contextvars.ContextVar("mx_refit_attributes", default={})
)


def configure(service_name: str) -> None:
    """Configure this process from standard OTLP HTTP environment variables."""
    global _configured_pid
    if _configured_pid == os.getpid():
        return
    traces = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    metrics = os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
    if not traces and not metrics:
        return
    from opentelemetry import metrics as otel_metrics
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource

    resource = Resource.create({"service.name": service_name})
    if traces:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=traces))
        )
        trace.set_tracer_provider(provider)
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


@contextlib.contextmanager
def span(name: str, attributes: Mapping[str, Any] | None = None) -> Iterator[Any]:
    if not enabled():
        yield _NoopSpan()
        return
    from opentelemetry import trace

    with trace.get_tracer("modelexpress.refit").start_as_current_span(name) as current:
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
        _value_histogram(name).record(int(value) if isinstance(value, bool) else value, dimensions)
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

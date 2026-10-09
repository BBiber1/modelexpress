# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional OpenTelemetry support for ModelExpress refits.

Callers use this module without importing OpenTelemetry directly. Installing the
``otel`` extra and setting OTLP endpoints enables export; otherwise it is inert.
"""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import logging
import os
import secrets
import threading
import time
from collections.abc import Iterator, Mapping, MutableMapping
from functools import lru_cache
from types import MappingProxyType
from typing import Any

_suppressed = contextvars.ContextVar("mx_telemetry_suppressed", default=False)
_configured_pid: int | None = None
_tracer: Any | None = None
_tracer_provider: Any | None = None
_meter_provider: Any | None = None
_refit_attributes: contextvars.ContextVar[Mapping[str, str | int]] = (
    contextvars.ContextVar("mx_refit_attributes", default=MappingProxyType({}))
)
_SHARED_ATTRIBUTES = frozenset(
    {
        "experiment",
        "step",
        "version_uid",
        "staging_mode",
        "refit.id",
        "refit.step",
        "refit.phase",
        "mx.experiment.run_id",
        "refit.aggregate",
    }
)
_ROLE_SPANS = frozenset(
    {
        "mx.refit.cycle",
        "mx.refit.trainers",
        "mx.refit.generators",
        "mx.refit.trainer",
        "mx.refit.generator",
        "mx.refit.orchestrator",
    }
)
_bounds: dict[tuple[int, int], list[int | None]] = {}
_bounds_lock = threading.RLock()


class RefitSpanProcessor:
    """Finalize role envelopes from local children and reported remote bounds."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        if span.name in _ROLE_SPANS:
            ctx = span.get_span_context()
            with _bounds_lock:
                _bounds[(ctx.trace_id, ctx.span_id)] = [None, None]
        self.processor.on_start(span, parent_context)

    def _on_ending(self, span: Any) -> None:
        self.processor._on_ending(span)

    def on_end(self, span: Any) -> None:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import ReadableSpan

        with _bounds_lock:
            interval = _bounds.pop((span.context.trace_id, span.context.span_id), None)
            bounded = interval is not None and interval[0] is not None
            if bounded or span.name == "mx.refit.cycle":
                span = ReadableSpan(
                    name=span.name,
                    context=span.context,
                    parent=span.parent,
                    resource=(
                        Resource(
                            {**span.resource.attributes, "service.name": "root"},
                            schema_url=span.resource.schema_url,
                        )
                        if span.name == "mx.refit.cycle"
                        else span.resource
                    ),
                    attributes=span.attributes,
                    events=span.events,
                    links=span.links,
                    kind=span.kind,
                    status=span.status,
                    start_time=interval[0] if bounded else span.start_time,
                    end_time=interval[1] if bounded else span.end_time,
                    instrumentation_scope=span.instrumentation_scope,
                )
            if span.parent is not None:
                parent = _bounds.get((span.context.trace_id, span.parent.span_id))
                if parent is not None:
                    _include_bounds(parent, [span.start_time, span.end_time])
        self.processor.on_end(span)

    def shutdown(self) -> None:
        self.processor.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.processor.force_flush(timeout_millis)


def _include_bounds(target: list, interval: list | tuple) -> None:
    start, end = interval
    if start is None or end is None or start > end:
        raise ValueError("Invalid refit interval")
    target[0] = min(target[0], start) if target[0] is not None else start
    target[1] = max(target[1], end) if target[1] is not None else end


def configure(service_name: str) -> None:
    """Configure this process from standard OTLP HTTP environment variables."""
    global _configured_pid, _tracer, _tracer_provider, _meter_provider
    if _configured_pid == os.getpid():
        return
    traces = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    metrics = os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
    if not traces and not metrics:
        return
    from opentelemetry.sdk.resources import Resource

    resource = Resource.create({"service.name": service_name})
    if traces:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

        class _ProcessIndependentIdGenerator(RandomIdGenerator):
            def generate_span_id(self) -> int:
                return secrets.randbits(64) or self.generate_span_id()

            def generate_trace_id(self) -> int:
                return secrets.randbits(128) or self.generate_trace_id()

        # vLLM seeds global random identically in its forked workers.
        provider = TracerProvider(
            resource=resource,
            id_generator=_ProcessIndependentIdGenerator(),
        )
        provider.add_span_processor(
            RefitSpanProcessor(BatchSpanProcessor(OTLPSpanExporter(endpoint=traces)))
        )
        _tracer_provider = provider
        atexit.register(provider.shutdown)
        _tracer = provider.get_tracer("modelexpress.refit")
    if metrics:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        _meter_provider = MeterProvider(
            metric_readers=[
                PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=metrics))
            ],
            resource=resource,
        )
        _histogram.cache_clear()
        _value_histogram.cache_clear()
        atexit.register(_meter_provider.shutdown)
    _configured_pid = os.getpid()


def enabled() -> bool:
    return bool(os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"))


@contextlib.contextmanager
def untraced() -> Iterator[None]:
    """Suppress bootstrap instrumentation until the existing RPC supplies a parent."""
    token = _suppressed.set(True)
    try:
        yield
    finally:
        _suppressed.reset(token)


def recording() -> bool:
    if not enabled() or _suppressed.get():
        return False
    from opentelemetry import trace

    current = trace.get_current_span()
    ctx = current.get_span_context()
    return not ctx.is_valid or bool(ctx.trace_flags.sampled)


def detailed() -> bool:
    from modelexpress import envs

    return envs.MX_REFIT_TRACE_DETAIL


_DETAIL_SPANS = frozenset(
    "mx.refit." + name
    for name in (
        "owner_validation",
        "vllm_load_layer",
        "vllm_load_batch",
        "install_commit",
        "materialization",
        "receive_copy",
        "post_load_processing",
        "transformation",
        "installation",
    )
)


def _process_tracer() -> Any:
    from opentelemetry import trace

    return (
        _tracer
        if _configured_pid == os.getpid() and _tracer is not None
        else trace.get_tracer("modelexpress.refit")
    )


class RefitCycle:
    """A native cycle or role envelope with an explicitly propagated parent."""

    def __init__(
        self,
        attributes: Mapping[str, Any] | None = None,
        *,
        name: str = "mx.refit.cycle",
        parent: Mapping[str, str] | None = None,
    ) -> None:
        self._span: Any | None = None
        self._attributes = {**_refit_attributes.get(), **(attributes or {})}
        self._interval: list[int | None] = [None, None]
        if not enabled():
            return
        from opentelemetry import context

        configure("modelexpress-rl")
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )

        ctx = (
            TraceContextTextMapPropagator().extract(parent)
            if parent is not None
            else (
                context.Context() if name == "mx.refit.cycle" else context.get_current()
            )
        )
        self._span = _process_tracer().start_span(name, context=ctx)
        if self.is_recording():
            self._span.set_attributes(self._attributes)
            span_ctx = self._span.get_span_context()
            with _bounds_lock:
                self._interval = _bounds.get(
                    (span_ctx.trace_id, span_ctx.span_id), self._interval
                )

    @property
    def interval(self) -> list[int] | None:
        return list(self._interval) if self._interval[0] is not None else None

    def include(self, interval: list[int] | None) -> None:
        if self.is_recording() and interval is not None:
            with _bounds_lock:
                _include_bounds(self._interval, interval)

    @contextlib.contextmanager
    def active(self) -> Iterator[RefitCycle]:
        if self._span is None:
            yield self
            return
        from opentelemetry import trace

        with trace.use_span(self._span, end_on_exit=False):
            yield self

    def is_recording(self) -> bool:
        return self._span is not None and self._span.is_recording()

    def inject(self, carrier: MutableMapping[str, str]) -> None:
        """Propagate this parent and its shared refit attributes."""
        if self._span is None:
            return
        from opentelemetry import trace
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )

        TraceContextTextMapPropagator().inject(
            carrier, context=trace.set_span_in_context(self._span)
        )
        _inject_attributes(carrier, self._attributes)

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        self._attributes.update(attributes)
        if self.is_recording():
            self._span.set_attributes(dict(attributes))

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
def refit_span(
    name: str,
    attributes: Mapping[str, Any] | None = None,
    *,
    parent: Mapping[str, str] | None = None,
) -> Iterator[RefitCycle]:
    cycle = RefitCycle(attributes, name=name, parent=parent)
    try:
        with cycle.active():
            yield cycle
    except BaseException as error:
        cycle.finish(error)
        raise
    else:
        cycle.finish()


@contextlib.contextmanager
def span(
    name: str,
    attributes: Mapping[str, Any] | None = None,
    *,
    start_time: int | None = None,
) -> Iterator[Any]:
    if not recording() or (name in _DETAIL_SPANS and not detailed()):
        yield _NoopSpan()
        return
    configure("modelexpress-rl")
    tracer = _process_tracer()
    with tracer.start_as_current_span(name, start_time=start_time) as current:
        if current.is_recording():
            current.set_attributes(dict(_refit_attributes.get()))
            if attributes:
                current.set_attributes(dict(attributes))
        yield current


def completed_span(
    name: str,
    start_time: int,
    end_time: int,
    attributes: Mapping[str, Any] | None = None,
) -> None:
    """Record a completed interval once its propagated parent is available."""
    if not recording():
        return
    configure("modelexpress-rl")
    current = _process_tracer().start_span(
        name,
        start_time=start_time,
        attributes={**_refit_attributes.get(), **(attributes or {})},
    )
    current.end(end_time=end_time)


def inject(carrier: MutableMapping[str, str]) -> None:
    """Write W3C traceparent and tracestate to an HTTP or gRPC carrier."""
    if not enabled() or _suppressed.get():
        return
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    TraceContextTextMapPropagator().inject(carrier)
    _inject_attributes(carrier, _refit_attributes.get())


def _inject_attributes(carrier: MutableMapping[str, str], attributes: Mapping) -> None:
    from opentelemetry import baggage, context
    from opentelemetry.baggage.propagation import W3CBaggagePropagator

    ctx = context.Context()
    for key in _SHARED_ATTRIBUTES & attributes.keys():
        ctx = baggage.set_baggage(key, str(attributes[key]), context=ctx)
    W3CBaggagePropagator().inject(carrier, context=ctx)


@contextlib.contextmanager
def extracted(carrier: Mapping[str, str]) -> Iterator[None]:
    """Make an incoming W3C parent current for the enclosed work."""
    if not enabled():
        yield
        return
    from opentelemetry import baggage, context
    from opentelemetry.baggage.propagation import W3CBaggagePropagator
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    shared = {
        key: int(value) if key in ("step", "refit.step") else value
        for key, value in baggage.get_all(
            W3CBaggagePropagator().extract(carrier)
        ).items()
        if key in _SHARED_ATTRIBUTES
    }
    token = context.attach(TraceContextTextMapPropagator().extract(carrier))
    try:
        with refit_attributes(shared):
            yield
    finally:
        context.detach(token)


def set_carrier_in_context(carrier: MutableMapping[str, str]) -> None:
    """Make an incoming W3C parent current for the enclosed work."""
    if not enabled():
        return
    from opentelemetry import context
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    _token = context.attach(TraceContextTextMapPropagator().extract(carrier))


@lru_cache(maxsize=16)
def _histogram(name: str) -> Any:
    from opentelemetry import metrics

    meter = (
        _meter_provider.get_meter("modelexpress.refit")
        if _meter_provider is not None
        else metrics.get_meter("modelexpress.refit")
    )
    return meter.create_histogram(name, unit="s")


def duration(name: str, seconds: float, attributes: Mapping[str, Any]) -> None:
    if os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"):
        _histogram(name).record(seconds, dict(attributes))


@lru_cache(maxsize=32)
def _value_histogram(name: str) -> Any:
    from opentelemetry import metrics

    unit = "By" if name.endswith("bytes") else "1"
    meter = (
        _meter_provider.get_meter("modelexpress.refit")
        if _meter_provider is not None
        else metrics.get_meter("modelexpress.refit")
    )
    return meter.create_histogram(f"mx_refit_{name}", unit=unit)


@contextlib.contextmanager
def refit_attributes(
    attributes: Mapping[str, str | int] | None = None,
    *,
    role: str | None = None,
    rank: int | None = None,
) -> Iterator[None]:
    """Apply role and rank to nested refit spans in the current context."""
    inherited = _refit_attributes.get()
    token = _refit_attributes.set(
        {
            **inherited,
            **(attributes or {}),
            **{key: inherited[key] for key in ("role", "rank") if key in inherited},
            **({"role": role} if role is not None else {}),
            **({"rank": rank} if rank is not None else {}),
        }
    )
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
        def intercept_unary_unary(self, continuation, details, request) -> grpc.Call | grpc.Future:
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
        if not recording():
            return None
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
        request = object()
        self._requests[request] = {
            "start_ns": time.time_ns(),
            "peer": peer,
            "end_ns": None,
            "complete": False,
            "native": None,
            "error": None,
        }
        return request

    def complete(self, request: Any, agent: Any, handle: Any) -> None:
        item = self._requests[request]
        item["end_ns"] = time.time_ns()
        item["complete"] = True
        try:
            sample = agent.get_xfer_telemetry(handle)
            item["native"] = {
                "total_bytes": int(sample.totalBytes),
                "desc_count": int(sample.descCount),
                "start_time_us": int(sample.startTime),
                "post_duration_s": int(sample.postDuration) / 1e6,
                "xfer_duration_s": int(sample.xferDuration) / 1e6,
            }
        except Exception as error:  # noqa: BLE001 - native telemetry is optional
            item["telemetry_error"] = str(error)

    def fail(self, request: Any, error: BaseException) -> None:
        self._error = error
        item = self._requests[request]
        if item["end_ns"] is None:
            item["end_ns"] = time.time_ns()
            item["error"] = error
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

        if detailed():
            for item in self._requests.values():
                request = _process_tracer().start_span(
                    "mx.refit.nixl_transfer",
                    context=trace.set_span_in_context(self._span),
                    start_time=item["start_ns"],
                    attributes={
                        **self._attributes,
                        "nixl.operation": "READ",
                        "nixl.remote_agent": item["peer"],
                        "nixl.complete": item["complete"],
                        "nixl.telemetry.available": item["native"] is not None,
                    },
                )
                if item["native"] is not None:
                    request.set_attributes(
                        {"nixl." + k: v for k, v in item["native"].items()}
                    )
                    with trace.use_span(request, end_on_exit=False):
                        self._metrics(item["native"], "request")
                if "telemetry_error" in item:
                    request.set_attribute(
                        "nixl.telemetry.error", item["telemetry_error"]
                    )
                if item["error"] is not None:
                    request.set_status(StatusCode.ERROR, str(item["error"]))
                    request.record_exception(item["error"])
                request.end(end_time=item["end_ns"])

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

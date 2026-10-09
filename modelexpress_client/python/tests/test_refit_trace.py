# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from modelexpress import telemetry
from modelexpress_rl.telemetry import RefitTrace


@pytest.fixture
def recording(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource({"service.name": "prime-rl-trainer"}))
    provider.add_span_processor(
        telemetry.RefitSpanProcessor(SimpleSpanProcessor(exporter))
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://unused/v1/traces")
    monkeypatch.setattr(telemetry, "_configured_pid", os.getpid())
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("refit-trace-test"))
    monkeypatch.setattr(telemetry, "_tracer_provider", provider)
    yield exporter
    provider.shutdown()


def _by_name(exporter):
    return {span.name: span for span in exporter.get_finished_spans()}


def _baggage(carrier):
    return dict(part.split("=", 1) for part in carrier["baggage"].split(","))


def test_refit_rank_prefers_active_rank_then_global_env_then_fallback(monkeypatch):
    monkeypatch.setenv("RANK", "8")
    assert telemetry.refit_rank(fallback=0) == 8

    with telemetry.refit_attributes({"role": "trainer", "rank": 3}):
        assert telemetry.refit_rank(fallback=0) == 3

    monkeypatch.delenv("RANK")
    assert telemetry.refit_rank(fallback=3) == 3
    assert telemetry.refit_rank() == 0


def test_trainer_e2e_span_and_timing_record_use_global_rank(
    recording, monkeypatch, caplog
):
    from modelexpress_rl.train.runtime import TrainerRuntime

    class Method:
        def stage(self, **_kwargs):
            return object()

        def publish(self, **_kwargs):
            pass

    class Runtime(TrainerRuntime):
        def _full_tensor(self):
            return self.method

    monkeypatch.setenv("RANK", "8")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MX_REFIT_TIMING", "1")
    runtime = Runtime(method=Method(), resources=None)
    runtime._bound_tensors = {"weight": object()}

    with caplog.at_level(logging.INFO, logger="modelexpress_rl.train.runtime"):
        runtime.publish_bound(version=SimpleNamespace(version_id="v1"))

    span = _by_name(recording)["mx.refit.trainer_refit_e2e"]
    assert span.attributes["rank"] == 8
    payloads = [
        json.loads(record.message.split(" ", 1)[1])
        for record in caplog.records
        if record.message.startswith("MX_REFIT_TIMING ")
    ]
    assert len(payloads) == 1
    assert payloads[0]["rank"] == 8


def test_rank_zero_trainer_owns_root_and_trainer_group(recording):
    trainer = RefitTrace.trainer(
        step=0, rank=0, trainers=4, generators=2, staging_mode="broadcast"
    )
    trainer.set_version("v7")
    carriers = trainer.context()

    assert _baggage(carriers["trainers"]) == {
        "step": "0",
        "refit.step": "0",
        "refit.phase": "cold",
        "staging_mode": "broadcast",
        "refit.aggregate": "True",
        "version_uid": "v7",
        "refit.id": "v7",
    }
    assert _baggage(carriers["root"]) == _baggage(carriers["trainers"])

    with trainer:
        assert not trace.get_current_span().get_span_context().is_valid
    spans = _by_name(recording)
    root = spans["mx.refit.cycle"]
    group = spans["mx.refit.trainers"]
    role = spans["mx.refit.trainer"]
    assert root.parent is None
    assert root.resource.attributes["service.name"] == "root"
    assert group.parent.span_id == root.context.span_id
    assert role.parent.span_id == group.context.span_id
    assert root.attributes["refit.expected_trainers"] == 4
    assert root.attributes["refit.expected_generators"] == 2
    assert role.attributes["role"] == "trainer"
    assert role.attributes["rank"] == 0
    assert role.attributes["step"] == role.attributes["refit.step"] == 0
    assert role.attributes["refit.phase"] == "cold"
    assert role.attributes["version_uid"] == role.attributes["refit.id"] == "v7"


def test_with_active_uses_role_and_generators_propagate_group(recording):
    trainer = RefitTrace.trainer(
        step=3, rank=0, trainers=1, generators=2, staging_mode="reshard"
    )
    trainer.set_version("v3")
    with trainer, trainer.active():
        with telemetry.span("trainer_work"):
            pass
        orchestrator = RefitTrace.orchestrator(step=3, staging_mode="reshard")
        orchestrator.bind(trainer.context()["root"], version_uid="v3")
        with orchestrator, orchestrator.generators() as carrier:
            assert _baggage(carrier)["version_uid"] == "v3"
            generator = RefitTrace.generator(version_uid="v3", rank=1, parent=carrier)
            with generator, generator.active(), telemetry.span("generate"):
                pass

    spans = _by_name(recording)
    assert (
        spans["trainer_work"].parent.span_id
        == spans["mx.refit.trainer"].context.span_id
    )
    assert (
        spans["mx.refit.generators"].parent.span_id
        == spans["mx.refit.cycle"].context.span_id
    )
    assert (
        spans["generate"].parent.span_id == spans["mx.refit.generator"].context.span_id
    )
    assert spans["generate"].attributes["role"] == "generator"
    assert spans["generate"].attributes["rank"] == 1
    assert spans["mx.refit.generators"].attributes["role"] == "generator"
    generator_role = spans["mx.refit.generator"]
    assert generator_role.parent.span_id == spans["mx.refit.generators"].context.span_id
    assert generator_role.attributes["role"] == "generator"
    assert generator_role.attributes["rank"] == 1
    assert generator_role.attributes["version_uid"] == "v3"
    assert generator_role.attributes["refit.phase"] == "warm"


def test_rank_one_trainer_defers_spans_until_broadcast_bind(recording):
    root = RefitTrace.trainer(
        step=8, rank=0, trainers=2, generators=0, staging_mode="gather"
    )
    root.set_version("v8")
    trainer = RefitTrace.trainer(
        step=8, rank=1, trainers=2, generators=0, staging_mode="gather"
    )
    with trainer.span("bootstrap", {"items": 3}):
        time.sleep(0.002)
    assert trainer._role is None
    assert len(trainer._pending) == 1

    trainer.bind(root.context()["trainers"], version_uid="v8")
    with trainer, trainer.active(), telemetry.span("after_bind"):
        pass
    root.finish()
    spans = _by_name(recording)
    role = next(
        span
        for span in recording.get_finished_spans()
        if span.name == "mx.refit.trainer" and span.attributes["rank"] == 1
    )
    bootstrap = spans["mx.refit.bootstrap"]
    assert bootstrap.parent.span_id == role.context.span_id
    assert bootstrap.attributes["items"] == 3
    assert bootstrap.attributes["role"] == "trainer"
    assert bootstrap.attributes["rank"] == 1
    assert bootstrap.attributes["step"] == 8
    assert bootstrap.attributes["version_uid"] == "v8"
    assert role.start_time <= bootstrap.start_time < bootstrap.end_time <= role.end_time
    assert spans["after_bind"].parent.span_id == role.context.span_id


def test_orchestrator_wait_span_binds_to_root_and_keeps_role_lifetime(recording):
    root = RefitTrace.trainer(
        step=2, rank=0, trainers=1, generators=0, staging_mode="broadcast"
    )
    root.set_version("v2")
    orchestrator = RefitTrace.orchestrator(step=2, staging_mode="broadcast")
    with orchestrator.span("wait_for_trainers"):
        time.sleep(0.002)
    assert not recording.get_finished_spans()

    orchestrator.bind(root.context()["root"], version_uid="v2")
    with orchestrator.active(), telemetry.span("publish"):
        pass
    time.sleep(0.002)
    with orchestrator.active(), telemetry.span("release"):
        pass
    orchestrator.finish()
    orchestrator.finish()
    root.finish()

    spans = _by_name(recording)
    role = spans["mx.refit.orchestrator"]
    assert role.parent.span_id == spans["mx.refit.cycle"].context.span_id
    assert role.attributes["role"] == "orchestrator"
    assert role.attributes["rank"] == 0
    assert role.attributes["step"] == 2
    assert role.attributes["version_uid"] == "v2"
    assert role.start_time <= spans["mx.refit.wait_for_trainers"].start_time
    assert spans["mx.refit.wait_for_trainers"].end_time < spans["publish"].start_time
    assert spans["publish"].end_time < spans["release"].start_time
    assert spans["release"].end_time <= role.end_time
    assert spans["mx.refit.wait_for_trainers"].parent.span_id == role.context.span_id
    assert (
        spans["publish"].parent.span_id
        == spans["release"].parent.span_id
        == role.context.span_id
    )


def test_failure_finishes_role_and_restores_ambient_context(recording):
    trainer = RefitTrace.trainer(
        step=1, rank=0, trainers=1, generators=0, staging_mode="broadcast"
    )
    error = RuntimeError("trainer failed")
    with telemetry.span("ambient") as ambient:
        ambient_context = ambient.get_span_context()
        with pytest.raises(RuntimeError, match="trainer failed"), trainer.active():
            raise error
        assert trace.get_current_span().get_span_context() == ambient_context
    assert trainer._pending == []
    spans = _by_name(recording)
    role = spans["mx.refit.trainer"]
    assert role.attributes["status"] == "failed"
    assert role.status.status_code == trace.StatusCode.ERROR
    assert role.events


def test_cancellation_clears_deferred_work_and_restores_context(recording):
    orchestrator = RefitTrace.orchestrator(step=4, staging_mode="broadcast")
    with telemetry.span("ambient") as ambient:
        ambient_context = ambient.get_span_context()
        with orchestrator.span("completed_before_cancel"):
            pass
        with pytest.raises(asyncio.CancelledError), orchestrator.span("cancelled"):
            raise asyncio.CancelledError
        assert trace.get_current_span().get_span_context() == ambient_context
    assert orchestrator._pending == []
    assert orchestrator._finished
    assert [span.name for span in recording.get_finished_spans()] == ["ambient"]


def test_disabled_trace_skips_exports_and_deferred_work(monkeypatch, recording):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    trainer = RefitTrace.trainer(
        step=0, rank=0, trainers=1, generators=0, staging_mode="broadcast"
    )
    with trainer, trainer.active(), trainer.span("work"):
        pass
    assert trainer._pending == []
    assert not trainer._role._span
    assert not recording.get_finished_spans()

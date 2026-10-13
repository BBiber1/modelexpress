# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logical-plan and physical-binding cache telemetry contracts."""

import hashlib
import json
import os
from collections.abc import Iterator
from types import MappingProxyType

import pytest

from tests.test_refit_bounded_descriptors import harness
pytest.importorskip("opentelemetry.sdk")

from modelexpress import telemetry
from modelexpress.refit.reshard.rendezvous import (
    PublishedShard,
    PublishedTensor,
    wrap_rendezvous_blob,
)
from modelexpress_rl.inference.adapter import TrainerSourceShard
from modelexpress_rl.inference.adapter import ReplicaGroup
from modelexpress_rl.inference.plan import TrainerSourceSnapshot
from modelexpress_rl.inference.plan import StreamingSettings
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def recording(monkeypatch) -> Iterator[InMemorySpanExporter]:
    monkeypatch.setenv("MX_REFIT_TRACE_DETAIL", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://unused/v1/traces")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", raising=False)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "_configured_pid", os.getpid())
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("logical-plan-test"))
    yield exporter
    provider.shutdown()


def _prepare(harness, *, worker="worker", generation=1, known_replicas=True):
    """Prepare with a controllable physical identity using the shared real harness."""
    from modelexpress_rl.inference import nixl_staged_transfer as module
    from modelexpress.refit.reshard.verify import tensor_digest
    from modelexpress import envs

    blobs = [wrap_rendezvous_blob(
        b"source", "source", "source:19000", [
            PublishedTensor(
                name=name, dtype="torch.float32", elsize=4, full_shape=(4, 4),
                shards=[PublishedShard(
                    agent_name="source", device_id=0, addr=tensor.data_ptr(),
                    shard_offset=(0, 0), shape=(4, 4),
                )],
            )
            for name, tensor in harness.sources.items()
        ],
    )]
    shard = TrainerSourceShard(
        "slot", worker, hashlib.sha256(blobs[0]).hexdigest(), "source:19000", blobs[0]
    )
    snapshot = TrainerSourceSnapshot(
        "mesh", generation, (shard,),
        replica_groups=(ReplicaGroup("slot", [shard]),) if known_replicas else (),
    )
    checksums = MappingProxyType({
        (name, "source", tensor.data_ptr(), (0, 0), (4, 4)): tensor_digest(tensor)
        for name, tensor in harness.sources.items()
    }) if envs.MX_RESHARD_PUBLISH_DIGEST else None
    return harness.transfer.prepare(
        trainer_snapshot=snapshot,
        manifests=blobs,
        capture_layout=lambda _manifest: (harness.capture, harness.layout),
        checksums=checksums,
    )


def _spans(exporter, name):
    return [span for span in exporter.get_finished_spans() if span.name == name]


def _prepare_in_generator_span(harness, **kwargs):
    with telemetry.span(
        "mx.refit.generator_update", {"role": "generator", "rank": 0}
    ):
        return _prepare(harness, **kwargs)


def test_cold_then_warm_plan_emits_one_snapshot_and_references(recording, harness, monkeypatch):
    cold = _prepare_in_generator_span(harness)
    cold_cached = harness.transfer._weight_update_plan

    def unexpected_logical_metadata(*_args, **_kwargs):
        pytest.fail("warm update must retain logical metadata")

    from modelexpress_rl.inference import nixl_staged_transfer as module
    from modelexpress.refit.reshard.transfer_plan import LogicalTransferPlan
    monkeypatch.setattr(module._NixlStagedTransfer, "_logical_metadata", staticmethod(unexpected_logical_metadata))
    for method in ("descriptor_count", "descriptor_savings", "extra_wire_bytes"):
        monkeypatch.setattr(LogicalTransferPlan, method, unexpected_logical_metadata)
    warm = _prepare_in_generator_span(harness)

    plans = _spans(recording, "mx.refit.logical_plan_create")
    rebindings = _spans(recording, "mx.refit.physical_replica_rebinding")
    assert len(plans) == 1
    assert len(rebindings) == 1
    assert plans[0].attributes["mx.measurement.logical_group_count"] == 1
    groups = json.loads(plans[0].attributes["mx.logical_plan.groups"])
    assert groups == [{"logical_shard_id": "slot", "replica_count": 1}]
    assert cold.metrics["plan_cache_misses"] == 1
    assert cold.metrics["plan_cache_hits"] == 0
    assert cold.metrics["physical_binding_cache_misses"] == 1
    assert warm.metrics["plan_cache_hits"] == 1
    assert warm.metrics["plan_cache_misses"] == 0
    assert warm.metrics["physical_binding_cache_hits"] == 1
    assert warm.metrics["physical_replica_cache_hits"] == 1
    assert warm.metrics["physical_replica_cache_misses"] == 0
    assert warm.metrics["physical_replica_rebind_attempts"] == 0
    cached = harness.transfer._weight_update_plan
    assert cached.logical_plan_id == cold_cached.logical_plan_id
    assert cached.physical_binding_id == cold_cached.physical_binding_id
    assert cached.telemetry_references["mx.logical_plan.id"] == cached.logical_plan_id
    assert cached.telemetry_references["mx.physical_binding.id"] == cached.physical_binding_id
    outer = _spans(recording, "mx.refit.generator_update")
    assert len(outer) == 2
    assert outer[-1].attributes["role"] == "generator"
    assert outer[-1].attributes["rank"] == 0
    assert outer[-1].attributes["mx.logical_plan.id"] == cached.logical_plan_id
    assert outer[-1].attributes["mx.physical_binding.id"] == cached.physical_binding_id


def test_physical_rebinding_keeps_logical_plan_and_reuses_prior_replica(recording, harness):
    a = _prepare(harness, worker="worker-A")
    b = _prepare(harness, worker="worker-B")
    a_again = _prepare(harness, worker="worker-A")

    assert len(_spans(recording, "mx.refit.logical_plan_create")) == 1
    assert len(_spans(recording, "mx.refit.physical_replica_rebinding")) == 3
    plans = [span.attributes["mx.logical_plan.id"] for span in _spans(recording, "mx.refit.logical_plan_create")]
    rebindings = _spans(recording, "mx.refit.physical_replica_rebinding")
    assert len(set(plans)) == 1
    assert len({span.attributes["mx.physical_binding.id"] for span in rebindings}) == 3
    assert b.metrics["physical_binding_cache_misses"] == 1
    assert a_again.metrics["physical_binding_cache_misses"] == 1
    assert a_again.metrics["physical_replica_cache_hits"] == 1
    assert a_again.metrics["physical_replica_cache_misses"] == 0
    last_groups = json.loads(rebindings[-1].attributes["mx.physical_binding.groups"])
    assert last_groups == [{
        "logical_shard_id": "slot", "worker_id": "worker-A", "remote_agent": "source"
    }]


def test_generation_change_creates_new_logical_plan_and_mesh_snapshot(recording, harness):
    first = _prepare(harness, generation=1)
    second = _prepare(harness, generation=2)

    plans = _spans(recording, "mx.refit.logical_plan_create")
    assert plans[0].attributes["mx.logical_plan.id"] != plans[1].attributes["mx.logical_plan.id"]
    assert [span.attributes["mx.logical_plan.mesh_generation"] for span in plans] == [1, 2]
    assert second.metrics["plan_cache_misses"] == 1
    assert second.metrics["physical_binding_cache_misses"] == 1


def test_unknown_replica_group_count_is_null(recording, harness):
    _prepare(harness, known_replicas=False)
    plan = _spans(recording, "mx.refit.logical_plan_create")[-1]
    groups = json.loads(plan.attributes["mx.logical_plan.groups"])
    assert groups == [{"logical_shard_id": "slot", "replica_count": None}]


@pytest.mark.parametrize(
    "harness", [StreamingSettings(512, "cpu")], indirect=True
)
def test_replica_cache_counters_count_slots_once_across_batches(recording, harness):
    prepared = _prepare(harness)
    assert len(prepared.batches) > 1
    assert prepared.metrics["physical_replica_cache_hits"] + prepared.metrics[
        "physical_replica_cache_misses"
    ] == 1
    assert prepared.metrics["physical_replica_cache_misses"] == 1

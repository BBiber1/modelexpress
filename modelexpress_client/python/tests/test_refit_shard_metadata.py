# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from collections.abc import Iterator
from concurrent import futures
from dataclasses import replace
from types import SimpleNamespace

import grpc
import pytest

from modelexpress.refit.timing import RefitTimingRecorder, use_refit_timing
import modelexpress_rl.inference.source.trainer as trainer_source_module

from modelexpress_rl import WeightPayloadFormat, refit_pb2, refit_pb2_grpc
from modelexpress_rl.control import WeightVersion, WeightVersionState
from modelexpress_rl.inference.source import TrainerSourceResolver
from tests.test_refit_generator_client import _RefitService, _STABLE_METADATA


class _Worker(refit_pb2_grpc.RefitWorkerServiceServicer):
    def __init__(self) -> None:
        self.blob = _STABLE_METADATA
        self.stable_requests = []
        self.version_requests = []
        self.metadata = refit_pb2.WeightVersionShardMetadata(
            version_id="version-a", worker_id="trainer-0", logical_shard_id="rank:0",
            stable_metadata_digest=hashlib.sha256(self.blob).hexdigest(),
            checksums=[refit_pb2.ShardChecksum(tensor_name="weight", digest="first")],
        )
        self.response_digest = ""
        self.blobs = {}
        self.clock = None
        self.stable_failures = 0
        self.version_failures = 0

    def GetTrainerShardMetadata(
        self, request: refit_pb2.GetTrainerShardMetadataRequest, context: grpc.ServicerContext,
    ) -> refit_pb2.GetTrainerShardMetadataResponse:
        self.stable_requests.append(request)
        if self.clock is not None:
            self.clock.now += 2
        if self.stable_failures:
            self.stable_failures -= 1
            context.abort(grpc.StatusCode.UNAVAILABLE, "stable metadata unavailable")
        blob = self.blobs.get(request.metadata_digest, self.blob)
        return refit_pb2.GetTrainerShardMetadataResponse(
            metadata=blob, metadata_digest=hashlib.sha256(blob).hexdigest(),
        )

    def GetWeightVersionShardMetadata(
        self, request: refit_pb2.GetWeightVersionShardMetadataRequest,
        context: grpc.ServicerContext,
    ) -> refit_pb2.GetWeightVersionShardMetadataResponse:
        self.version_requests.append(request)
        if self.clock is not None:
            self.clock.now += 3
        if self.version_failures:
            self.version_failures -= 1
            context.abort(grpc.StatusCode.UNAVAILABLE, "version metadata unavailable")
        return refit_pb2.GetWeightVersionShardMetadataResponse(
            metadata=self.metadata, version_metadata_digest=self.response_digest,
        )


def _advertise(case: SimpleNamespace) -> None:
    case.control.shards[0].stable_metadata_digest = hashlib.sha256(case.worker.blob).hexdigest()
    digest = hashlib.sha256(
        case.worker.metadata.SerializeToString(deterministic=True)
    ).hexdigest()
    case.worker.response_digest = digest
    case.control.shards[0].version_metadata_digest = digest


@pytest.fixture
def protocol(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    port = server.add_insecure_port("127.0.0.1:0")
    control = _RefitService(endpoint=f"127.0.0.1:{port}")
    control.shards = control.shards[:1]
    worker = _Worker()
    refit_pb2_grpc.add_RefitServiceServicer_to_server(control, server)
    refit_pb2_grpc.add_RefitWorkerServiceServicer_to_server(worker, server)
    server.start()
    with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
        service = refit_pb2_grpc.RefitServiceStub(channel)
        case = SimpleNamespace(
            control=control, worker=worker, cached_snapshot=None,
            version=WeightVersion(
                version_id="version-a", model_name="test/model",
                payload_format=WeightPayloadFormat.FULL_TENSOR, layout_signature="layout-a",
                state=WeightVersionState.READY, created_at_unix_ms=0,
                trainer_mesh_generation=1, trainer_mesh_id="mesh-a",
            ),
        )
        case.resolver = TrainerSourceResolver(
            service=lambda: service, rpc_timeout_seconds=2,
            cached_source=lambda: case.cached_snapshot,
        )
        _advertise(case)
        try:
            yield case
        finally:
            server.stop(grace=None).wait()


@pytest.mark.parametrize("field", [
    "version_id", "worker_id", "logical_shard_id", "stable_metadata_digest",
])
def test_version_record_identity_mismatch_has_no_transfer_candidate(
    protocol: SimpleNamespace, field: str,
) -> None:
    setattr(protocol.worker.metadata, field, "different")
    _advertise(protocol)
    assert list(protocol.resolver.candidates(protocol.version)) == []
    assert len(protocol.worker.version_requests) == 1


@pytest.mark.parametrize("corrupt", ["advertised_hash", "response_hash", "record_bytes"])
def test_wrong_version_record_hash_has_no_transfer_candidate(
    protocol: SimpleNamespace, corrupt: str,
) -> None:
    if corrupt == "advertised_hash":
        protocol.control.shards[0].version_metadata_digest = "wrong"
    elif corrupt == "response_hash":
        protocol.worker.response_digest = "wrong"
    else:
        protocol.worker.metadata.checksums[0].digest = "changed-without-updating-hash"
    assert list(protocol.resolver.candidates(protocol.version)) == []


@pytest.mark.parametrize("corrupt", ["advertised_hash", "response_bytes"])
def test_wrong_stable_metadata_hash_is_rejected_before_version_fetch(
    protocol: SimpleNamespace, corrupt: str,
) -> None:
    if corrupt == "advertised_hash":
        protocol.control.shards[0].stable_metadata_digest = "wrong"
    else:
        payload = json.loads(protocol.worker.blob)
        payload["metadata_endpoint"] = "another-trainer:19000"
        protocol.worker.blob = json.dumps(payload).encode()
    assert list(protocol.resolver.candidates(protocol.version)) == []
    assert protocol.worker.version_requests == []


@pytest.mark.parametrize("rows", ["duplicate", "unknown", "missing", "empty_digest"])
def test_invalid_checksum_coverage_has_no_transfer_candidate(
    protocol: SimpleNamespace, rows: str,
) -> None:
    checksums = protocol.worker.metadata.checksums
    if rows == "duplicate":
        checksums.add().CopyFrom(checksums[0])
    elif rows == "unknown":
        checksums[0].tensor_name = "unknown"
    elif rows == "missing":
        del checksums[:]
    else:
        checksums[0].digest = ""
    _advertise(protocol)
    assert list(protocol.resolver.candidates(protocol.version)) == []


@pytest.mark.parametrize(("field", "value"), [
    ("tensor_count", True), ("tensor_count", 2), ("tensor_count", 0),
    ("total_bytes", True), ("total_bytes", 9), ("total_bytes", 0),
])
def test_invalid_stable_counts_are_rejected_before_version_fetch(
    protocol: SimpleNamespace, field: str, value: int | bool,
) -> None:
    payload = json.loads(protocol.worker.blob)
    payload[field] = value
    protocol.worker.blob = json.dumps(payload).encode()
    _advertise(protocol)
    assert list(protocol.resolver.candidates(protocol.version)) == []
    assert protocol.worker.version_requests == []


def test_duplicate_physical_checksum_keys_are_rejected_before_version_fetch(
    protocol: SimpleNamespace,
) -> None:
    payload = json.loads(protocol.worker.blob)
    payload["tensors"][0]["shards"].append(dict(payload["tensors"][0]["shards"][0]))
    payload["total_bytes"] = 16
    protocol.worker.blob = json.dumps(payload).encode()
    _advertise(protocol)
    assert list(protocol.resolver.candidates(protocol.version)) == []
    assert protocol.worker.version_requests == []


def test_coverage_only_metadata_cannot_resolve_physical_reads(protocol: SimpleNamespace) -> None:
    protocol.worker.blob = json.dumps({"tensors": [{
        "name": "weight", "dtype": "torch.bfloat16", "elsize": 2,
        "full_shape": [4], "shards": [{"shard_offset": [0], "shape": [4]}],
    }]}).encode()
    _advertise(protocol)
    assert list(protocol.resolver.candidates(protocol.version)) == []
    assert protocol.worker.version_requests == []


def test_verification_disabled_resolves_without_version_rpc(
    protocol: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    protocol.control.shards[0].version_metadata_digest = ""
    resolved, = protocol.resolver.candidates(protocol.version)
    assert resolved.snapshot.shards[0].metadata == protocol.worker.blob
    assert dict(resolved.checksums) == {}
    assert len(protocol.worker.stable_requests) == 1
    assert protocol.worker.version_requests == []


def _next_version(protocol: SimpleNamespace) -> WeightVersion:
    protocol.worker.metadata.version_id = "version-b"
    protocol.worker.metadata.checksums[0].digest = "second"
    protocol.control.shards[0].version_id = "version-b"
    protocol.control.version.uid = "version-b"
    _advertise(protocol)
    return replace(protocol.version, version_id="version-b")


@pytest.mark.parametrize("verify", [False, True])
def test_new_version_resolves_fresh_checksums_without_refetching_stable_metadata(
    protocol: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, verify: bool,
) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", str(int(verify)))
    first, = protocol.resolver.candidates(protocol.version)
    protocol.cached_snapshot = first.snapshot
    second, = protocol.resolver.candidates(_next_version(protocol))
    assert first.snapshot == second.snapshot
    assert list(first.checksums.values()) == (["first"] if verify else [])
    assert list(second.checksums.values()) == (["second"] if verify else [])
    assert len(protocol.worker.stable_requests) == 1
    assert protocol.control.list_calls == 2
    assert protocol.control.mesh_calls == (4 if verify else 3)
    assert [request.version_id for request in protocol.worker.version_requests] == (
        ["version-a", "version-b"] if verify else []
    )


def test_cached_selection_missing_worker_uses_current_publication_replica(
    protocol: SimpleNamespace,
) -> None:
    first, = protocol.resolver.candidates(protocol.version)
    protocol.cached_snapshot = first.snapshot
    protocol.control.shards[0].worker_id = "trainer-replacement"
    protocol.worker.metadata.worker_id = "trainer-replacement"
    payload = json.loads(protocol.worker.blob)
    payload["tensors"][0]["shards"][0]["addr"] = 8192
    protocol.worker.blob = json.dumps(payload).encode()
    protocol.worker.metadata.stable_metadata_digest = hashlib.sha256(protocol.worker.blob).hexdigest()
    second, = protocol.resolver.candidates(_next_version(protocol))
    assert second.snapshot.shards[0].worker_id == "trainer-replacement"
    assert list(second.checksums.values()) == ["second"]
    assert len(protocol.worker.stable_requests) == 2
    assert protocol.control.list_calls == 2
    assert protocol.control.mesh_calls == 4


def test_new_mesh_generation_can_replace_cached_workers_stable_storage(
    protocol: SimpleNamespace,
) -> None:
    first, = protocol.resolver.candidates(protocol.version)
    protocol.cached_snapshot = first.snapshot
    payload = json.loads(protocol.worker.blob)
    payload["tensors"][0]["shards"][0]["addr"] = 8192
    protocol.worker.blob = json.dumps(payload).encode()
    protocol.worker.metadata.stable_metadata_digest = hashlib.sha256(protocol.worker.blob).hexdigest()
    protocol.control.mesh_generation_on_recheck = 2
    requested = replace(_next_version(protocol), trainer_mesh_generation=2)
    second, = protocol.resolver.candidates(requested)
    assert second.snapshot.mesh_generation == 2
    assert second.snapshot.shards[0].worker_id == first.snapshot.shards[0].worker_id
    assert second.snapshot.shards[0].metadata == protocol.worker.blob
    assert second.snapshot.shards[0].stable_metadata_digest != first.snapshot.shards[0].stable_metadata_digest
    assert list(second.checksums.values()) == ["second"]
    assert next(iter(second.checksums))[2] == 8192
    assert len(protocol.worker.stable_requests) == 2
    assert protocol.control.list_calls == 2
    assert protocol.control.mesh_calls == 4


def test_requested_version_generation_mismatch_rejects_cached_selection(
    protocol: SimpleNamespace,
) -> None:
    first, = protocol.resolver.candidates(protocol.version)
    protocol.cached_snapshot = first.snapshot
    protocol.control.mesh_generation_on_recheck = 2
    with pytest.raises(RuntimeError, match="generation"):
        next(protocol.resolver.candidates(_next_version(protocol)))
    assert protocol.control.list_calls == 1
    assert len(protocol.worker.stable_requests) == 1
    assert len(protocol.worker.version_requests) == 1


@pytest.fixture
def discovery_timing(
    protocol: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    clock = SimpleNamespace(now=0.0)
    protocol.worker.clock = clock
    read_clock = lambda: clock.now
    monkeypatch.setattr(trainer_source_module, "perf_counter", read_clock)
    recorder = RefitTimingRecorder(backend="test", version="version-a", clock=read_clock)
    return SimpleNamespace(clock=clock, recorder=recorder)


@pytest.mark.parametrize("verify", [False, True])
def test_discovery_timing_accumulates_cold_and_warm_rpc_work(
    protocol: SimpleNamespace, discovery_timing: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch, verify: bool,
) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", str(int(verify)))
    with use_refit_timing(discovery_timing.recorder):
        first_candidates = protocol.resolver.candidates(protocol.version)
        first = next(first_candidates)
        first_candidates.close()
        protocol.cached_snapshot = first.snapshot
        second_candidates = protocol.resolver.candidates(_next_version(protocol))
        second = next(second_candidates)
        before_close = discovery_timing.recorder.as_dict()["stages"]["source_preparation"]
        discovery_timing.clock.now += 100
        second_candidates.close()
    stage = discovery_timing.recorder.as_dict()["stages"]["source_preparation"]
    assert stage == before_close
    metrics = stage["metadata"]
    assert metrics["manifest_list_count"] == protocol.control.list_calls == 2
    assert metrics["published_shard_count"] == 2
    assert metrics["manifest_cache_hits"] == 1
    assert metrics["manifest_cache_misses"] == 1
    assert metrics["manifest_fetch_count"] == len(protocol.worker.stable_requests) == 1
    assert metrics["manifest_fetch_bytes"] == len(protocol.worker.blob)
    assert metrics["manifest_bytes"] == 2 * len(protocol.worker.blob)
    assert metrics.get("version_metadata_fetch_count", 0) == len(protocol.worker.version_requests) == (2 if verify else 0)
    assert list(second.checksums.values()) == (["second"] if verify else [])
    assert metrics["manifest_fetch_s"] == 2
    assert stage["duration_ms"] == (8000 if verify else 2000)


@pytest.mark.parametrize("failure", ["stable", "version"])
def test_discovery_timing_retains_partial_failed_rpc_work_before_fallback(
    protocol: SimpleNamespace, discovery_timing: SimpleNamespace, failure: str,
) -> None:
    alternative = refit_pb2.WeightVersionShard()
    alternative.CopyFrom(protocol.control.shards[0])
    alternative.worker_id = "trainer-b"
    protocol.worker.metadata.worker_id = "trainer-b"
    _advertise(protocol)
    alternative.version_metadata_digest = protocol.control.shards[0].version_metadata_digest
    protocol.control.shards.append(alternative)
    if failure == "stable":
        protocol.worker.stable_failures = 1
    else:
        protocol.worker.version_failures = 1
    with use_refit_timing(discovery_timing.recorder):
        candidates = protocol.resolver.candidates(protocol.version)
        resolved = next(candidates)
        before_close = discovery_timing.recorder.as_dict()["stages"]["source_preparation"]
        candidates.close()
    stage = discovery_timing.recorder.as_dict()["stages"]["source_preparation"]
    assert stage == before_close
    assert resolved.snapshot.shards[0].worker_id == "trainer-b"
    assert list(resolved.checksums.values()) == ["first"]
    metrics = stage["metadata"]
    assert metrics["manifest_list_count"] == protocol.control.list_calls == 1
    assert metrics["published_shard_count"] == 2
    assert metrics["manifest_cache_misses"] == 2
    assert metrics.get("manifest_cache_hits", 0) == 0
    successful_stable_fetches = 1 if failure == "stable" else 2
    assert len(protocol.worker.stable_requests) == 2
    assert metrics["manifest_fetch_count"] == successful_stable_fetches
    assert metrics["manifest_fetch_bytes"] == successful_stable_fetches * len(protocol.worker.blob)
    assert metrics["version_metadata_fetch_count"] == 1
    assert len(protocol.worker.version_requests) == (1 if failure == "stable" else 2)
    assert metrics["manifest_bytes"] == len(protocol.worker.blob)
    assert metrics["manifest_fetch_s"] == 4
    assert stage["duration_ms"] == (7000 if failure == "stable" else 10000)


@pytest.mark.parametrize("resume", [False, True])
def test_lazy_discovery_timing_excludes_consumer_pause_and_flushes_once(
    protocol: SimpleNamespace, discovery_timing: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch, resume: bool,
) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    alternative = refit_pb2.WeightVersionShard()
    alternative.CopyFrom(protocol.control.shards[0])
    alternative.worker_id = "trainer-b"
    payload = json.loads(protocol.worker.blob)
    payload["tensors"][0]["shards"][0]["addr"] = 8192
    alternative_blob = json.dumps(payload).encode()
    alternative.stable_metadata_digest = hashlib.sha256(alternative_blob).hexdigest()
    protocol.worker.blobs[alternative.stable_metadata_digest] = alternative_blob
    protocol.control.shards.append(alternative)
    with use_refit_timing(discovery_timing.recorder):
        candidates = protocol.resolver.candidates(protocol.version)
        first = next(candidates)
        first_stage = discovery_timing.recorder.as_dict()["stages"]["source_preparation"]
        assert first.snapshot.shards[0].worker_id == "trainer-0"
        assert first_stage["metadata"]["manifest_fetch_count"] == 1
        discovery_timing.clock.now += 100
        if resume:
            second = next(candidates)
            assert second.snapshot.shards[0].worker_id == "trainer-b"
        before_close = discovery_timing.recorder.as_dict()["stages"]["source_preparation"]
        discovery_timing.clock.now += 100
        candidates.close()
    stage = discovery_timing.recorder.as_dict()["stages"]["source_preparation"]
    assert stage == before_close
    metrics = stage["metadata"]
    resolved_count = 2 if resume else 1
    assert metrics["manifest_list_count"] == protocol.control.list_calls == 1
    assert metrics["published_shard_count"] == 2
    assert metrics["manifest_fetch_count"] == len(protocol.worker.stable_requests) == resolved_count
    assert metrics["manifest_cache_misses"] == resolved_count
    assert metrics.get("manifest_cache_hits", 0) == 0
    assert metrics["manifest_fetch_bytes"] == len(protocol.worker.blob) + (len(alternative_blob) if resume else 0)
    assert metrics["manifest_bytes"] == metrics["manifest_fetch_bytes"]
    assert metrics["manifest_fetch_s"] == resolved_count * 2
    assert stage["duration_ms"] == resolved_count * 2000
    assert protocol.worker.version_requests == []


@pytest.mark.parametrize("failure", ["stable", "version"])
def test_exhausted_discovery_flushes_failed_rpc_timing(
    protocol: SimpleNamespace, discovery_timing: SimpleNamespace, failure: str,
) -> None:
    if failure == "stable":
        protocol.worker.stable_failures = 1
    else:
        protocol.worker.version_failures = 1
    with use_refit_timing(discovery_timing.recorder):
        assert list(protocol.resolver.candidates(protocol.version)) == []
    stage = discovery_timing.recorder.as_dict()["stages"]["source_preparation"]
    metrics = stage["metadata"]
    assert metrics["manifest_list_count"] == protocol.control.list_calls == 1
    assert metrics["published_shard_count"] == 1
    assert metrics["manifest_cache_misses"] == 1
    assert metrics.get("manifest_fetch_count", 0) == (0 if failure == "stable" else 1)
    assert metrics.get("manifest_fetch_bytes", 0) == (0 if failure == "stable" else len(protocol.worker.blob))
    assert metrics.get("manifest_bytes", 0) == 0
    assert metrics.get("version_metadata_fetch_count", 0) == 0
    assert len(protocol.worker.stable_requests) == 1
    assert len(protocol.worker.version_requests) == (0 if failure == "stable" else 1)
    assert metrics["manifest_fetch_s"] == 2
    assert stage["duration_ms"] == (2000 if failure == "stable" else 5000)

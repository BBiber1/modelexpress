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

from modelexpress_rl import WeightPayloadFormat, refit_pb2, refit_pb2_grpc
from modelexpress_rl.control import WeightVersion, WeightVersionState
from modelexpress_rl.inference.source import TrainerSourceResolver
from modelexpress_rl.inference.plan import ResolvedTrainerSource
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

    def GetTrainerShardMetadata(
        self, request: refit_pb2.GetTrainerShardMetadataRequest, context: grpc.ServicerContext,
    ) -> refit_pb2.GetTrainerShardMetadataResponse:
        self.stable_requests.append(request)
        return refit_pb2.GetTrainerShardMetadataResponse(
            metadata=self.blob, metadata_digest=hashlib.sha256(self.blob).hexdigest(),
        )

    def GetWeightVersionShardMetadata(
        self, request: refit_pb2.GetWeightVersionShardMetadataRequest,
        context: grpc.ServicerContext,
    ) -> refit_pb2.GetWeightVersionShardMetadataResponse:
        self.version_requests.append(request)
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
            learned={}, learned_scope=None,
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
            cached_replicas=lambda mesh_id, generation: (
                tuple(case.learned.values())
                if case.learned_scope == (mesh_id, generation) else ()
            ),
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


def test_duplicate_geometry_with_distinct_addresses_rejects_before_version_fetch(
    protocol: SimpleNamespace,
) -> None:
    payload = json.loads(protocol.worker.blob)
    shards = payload["tensors"][0]["shards"]
    duplicate = dict(shards[0])
    duplicate["addr"] = 8192
    shards.append(duplicate)
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


def _next_version(
    protocol: SimpleNamespace, version_id: str = "version-b", digest: str = "second",
) -> WeightVersion:
    protocol.worker.metadata.version_id = version_id
    protocol.worker.metadata.checksums[0].digest = digest
    protocol.control.shards[0].version_id = version_id
    protocol.control.version.uid = version_id
    _advertise(protocol)
    return replace(protocol.version, version_id=version_id)


def _remember(protocol: SimpleNamespace, resolved: ResolvedTrainerSource) -> None:
    protocol.cached_snapshot = resolved.snapshot
    scope = (resolved.snapshot.mesh_id, resolved.snapshot.mesh_generation)
    if protocol.learned_scope != scope:
        protocol.learned.clear()
    protocol.learned_scope = scope
    protocol.learned.update({
        (shard.source_slot_id, shard.worker_id): shard
        for shard in resolved.snapshot.shards
    })


@pytest.mark.parametrize("verify", [False, True])
def test_new_version_resolves_fresh_checksums_without_refetching_stable_metadata(
    protocol: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, verify: bool,
) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", str(int(verify)))
    first, = protocol.resolver.candidates(protocol.version)
    _remember(protocol, first)
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
    _remember(protocol, first)
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
    _remember(protocol, first)
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
    _remember(protocol, first)
    protocol.control.mesh_generation_on_recheck = 2
    with pytest.raises(RuntimeError, match="generation"):
        next(protocol.resolver.candidates(_next_version(protocol)))
    assert protocol.control.list_calls == 1
    assert len(protocol.worker.stable_requests) == 1
    assert len(protocol.worker.version_requests) == 1


@pytest.mark.parametrize("verify", [False, True])
def test_learned_replicas_reuse_stable_metadata_across_a_b_a_selection(
    protocol: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, verify: bool,
) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", str(int(verify)))
    first, = protocol.resolver.candidates(protocol.version)
    _remember(protocol, first)
    original_blob = protocol.worker.blob
    protocol.control.shards[0].worker_id = "trainer-b"
    protocol.worker.metadata.worker_id = "trainer-b"
    payload = json.loads(original_blob)
    payload["tensors"][0]["shards"][0]["addr"] = 8192
    protocol.worker.blob = json.dumps(payload).encode()
    protocol.worker.metadata.stable_metadata_digest = hashlib.sha256(protocol.worker.blob).hexdigest()
    second, = protocol.resolver.candidates(_next_version(protocol))
    _remember(protocol, second)
    protocol.control.shards[0].worker_id = "trainer-0"
    protocol.worker.metadata.worker_id = "trainer-0"
    protocol.worker.blob = original_blob
    protocol.worker.metadata.stable_metadata_digest = hashlib.sha256(original_blob).hexdigest()
    third, = protocol.resolver.candidates(_next_version(protocol, "version-c", "third"))
    assert [resolved.snapshot.shards[0].worker_id for resolved in (first, second, third)] == [
        "trainer-0", "trainer-b", "trainer-0",
    ]
    assert third.snapshot.shards[0].metadata == original_blob
    assert [list(resolved.checksums.values()) for resolved in (first, second, third)] == (
        [["first"], ["second"], ["third"]] if verify else [[], [], []]
    )
    assert len(protocol.worker.stable_requests) == 2
    assert [request.version_id for request in protocol.worker.version_requests] == (
        ["version-a", "version-b", "version-c"] if verify else []
    )
    assert protocol.control.list_calls == 3


@pytest.mark.parametrize("drift", ["digest", "endpoint"])
@pytest.mark.parametrize("selected", [False, True])
def test_known_replica_physical_drift_rejects_before_worker_rpc(
    protocol: SimpleNamespace, drift: str, selected: bool,
) -> None:
    first, = protocol.resolver.candidates(protocol.version)
    _remember(protocol, first)
    original = refit_pb2.WeightVersionShard()
    original.CopyFrom(protocol.control.shards[0])
    protocol.control.shards[0].worker_id = "trainer-b"
    protocol.worker.metadata.worker_id = "trainer-b"
    second, = protocol.resolver.candidates(_next_version(protocol))
    _remember(protocol, second)
    requested = _next_version(protocol, "version-c", "third")
    original.version_id = "version-c"
    if drift == "digest":
        original.stable_metadata_digest = "changed-known-storage"
    else:
        original.metadata_endpoint = "different-worker:19000"
    if selected:
        protocol.control.shards = [original]
    else:
        protocol.control.shards.append(original)
    stable_calls = len(protocol.worker.stable_requests)
    version_calls = len(protocol.worker.version_requests)
    with pytest.raises(RuntimeError):
        next(protocol.resolver.candidates(requested))
    assert len(protocol.worker.stable_requests) == stable_calls
    assert len(protocol.worker.version_requests) == version_calls
    assert protocol.control.list_calls == 3

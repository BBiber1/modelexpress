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
            control=control, worker=worker,
            resolver=TrainerSourceResolver(service=lambda: service, rpc_timeout_seconds=2),
            version=WeightVersion(
                version_id="version-a", model_name="test/model",
                payload_format=WeightPayloadFormat.FULL_TENSOR, layout_signature="layout-a",
                state=WeightVersionState.READY, created_at_unix_ms=0,
                trainer_mesh_generation=1, trainer_mesh_id="mesh-a",
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


def test_new_version_resolves_fresh_checksums_without_refetching_stable_metadata(
    protocol: SimpleNamespace,
) -> None:
    first, = protocol.resolver.candidates(protocol.version)
    protocol.worker.metadata.version_id = "version-b"
    protocol.worker.metadata.checksums[0].digest = "second"
    protocol.control.shards[0].version_id = "version-b"
    protocol.control.version.uid = "version-b"
    _advertise(protocol)
    second, = protocol.resolver.candidates(replace(protocol.version, version_id="version-b"))
    assert first.snapshot == second.snapshot
    assert list(first.checksums.values()) == ["first"]
    assert list(second.checksums.values()) == ["second"]
    assert len(protocol.worker.stable_requests) == 1
    assert [request.version_id for request in protocol.worker.version_requests] == [
        "version-a", "version-b",
    ]

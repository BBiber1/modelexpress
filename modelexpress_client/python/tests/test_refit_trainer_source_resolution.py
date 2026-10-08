# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import hashlib
from types import SimpleNamespace

import pytest
from modelexpress.refit.timing import RefitTimingRecorder, use_refit_timing
from modelexpress_rl import refit_pb2
from modelexpress_rl.inference.source import trainer
from modelexpress_rl.train import WeightPayloadFormat


def sources(monkeypatch, count, *, miss=None, failed_replica=False) -> tuple[trainer.TrainerSourceResolver, SimpleNamespace, list[str], int]:
    manifest = b'{"tensors": []}'
    digest = hashlib.sha256(manifest).hexdigest()
    workers, shards = {}, []
    for index in range(count):
        worker, slot = f"worker-{index}", f"slot-{index}"
        workers[worker] = refit_pb2.TrainerTensorsMetadata(logical_shard_id=slot)
        shards.append(
            refit_pb2.WeightVersionShard(
                version_id="v1",
                logical_shard_id=slot,
                worker_id=worker,
                manifest_endpoint=worker,
                manifest_digest=digest,
            )
        )
    if failed_replica:
        workers["a-failed"] = refit_pb2.TrainerTensorsMetadata(
            logical_shard_id="slot-0"
        )
        shards.append(
            refit_pb2.WeightVersionShard(
                version_id="v1",
                logical_shard_id="slot-0",
                worker_id="a-failed",
                manifest_endpoint="a-failed",
                manifest_digest=digest,
            )
        )
    mesh = refit_pb2.TrainerMesh(mesh_id="mesh", generation=1, workers=workers)
    service = SimpleNamespace(
        GetTrainerMesh=lambda *args, **kwargs: refit_pb2.GetTrainerMeshResponse(
            mesh=mesh
        ),
        ListWeightVersionShards=lambda *args, **kwargs: (
            refit_pb2.ListWeightVersionShardsResponse(shards=shards)
        ),
    )
    resolver = trainer.TrainerSourceResolver(
        service=lambda: service, rpc_timeout_seconds=1
    )
    for shard in shards:
        if shard.worker_id != miss and shard.worker_id != "a-failed":
            resolver._manifest_cache[(shard.logical_shard_id, shard.worker_id)] = (
                shard.manifest_endpoint,
                digest,
                manifest,
                "structure",
            )
    fetched = []

    def stub(endpoint) -> SimpleNamespace:
        def fetch(*args, **kwargs) -> refit_pb2.GetWeightVersionShardManifestResponse:
            fetched.append(endpoint)
            if endpoint == "a-failed":
                raise RuntimeError("injected fetch failure")
            return refit_pb2.GetWeightVersionShardManifestResponse(
                manifest=manifest, manifest_digest=digest
            )

        return SimpleNamespace(GetWeightVersionShardManifest=fetch)

    monkeypatch.setattr(
        trainer.grpc,
        "insecure_channel",
        lambda endpoint, **kwargs: contextlib.nullcontext(endpoint),
    )
    monkeypatch.setattr(trainer.refit_pb2_grpc, "RefitWorkerServiceStub", stub)
    version = SimpleNamespace(
        version_id="v1",
        trainer_mesh_id="mesh",
        base_version_id=None,
        layout_signature="layout",
        payload_format=WeightPayloadFormat.FULL_TENSOR,
    )
    return resolver, version, fetched, len(manifest)


@pytest.mark.parametrize("count", [1, 256])
def test_warm_shards_share_one_resolution_measurement(monkeypatch, count) -> None:
    resolver, version, fetched, size = sources(monkeypatch, count)
    recorder = RefitTimingRecorder(backend="rl_generator", version="v1", rank=0)
    with use_refit_timing(recorder):
        candidates = resolver.candidates(version)
        assert len(next(candidates).inputs.sources) == count
        stage = recorder.as_dict()["stages"]["source_preparation"]
        assert stage["count"] == 2
        assert stage["metadata"]["manifest_cache_hits"] == count
        assert stage["metadata"]["manifest_cache_misses"] == 0
        assert stage["metadata"]["manifest_bytes"] == count * size
        assert "source_resolution_s" in stage["metadata"]
        before = dict(stage["metadata"])
        candidates.close()
        assert recorder.as_dict()["stages"]["source_preparation"]["metadata"] == before
    assert fetched == []


def test_mixed_shards_count_failed_fetches_and_successful_bytes(monkeypatch) -> None:
    resolver, version, fetched, size = sources(
        monkeypatch, 3, miss="worker-1", failed_replica=True
    )
    recorder = RefitTimingRecorder(backend="rl_generator", version="v1", rank=0)
    with use_refit_timing(recorder):
        candidates = resolver.candidates(version)
        assert len(next(candidates).inputs.sources) == 3
        candidates.close()
    metadata = recorder.as_dict()["stages"]["source_preparation"]["metadata"]
    assert metadata["manifest_cache_hits"] == 2
    assert metadata["manifest_cache_misses"] == 2
    assert metadata["manifest_fetch_count"] == 2
    assert metadata["manifest_fetch_bytes"] == size
    assert metadata["manifest_bytes"] == 3 * size
    assert "manifest_fetch_s" in metadata
    assert "manifest_hash_s" in metadata
    assert "manifest_fingerprint_s" in metadata
    assert fetched == ["a-failed", "worker-1"]


def test_warm_resolution_without_timing(monkeypatch) -> None:
    resolver, version, fetched, _ = sources(monkeypatch, 256)
    assert len(next(resolver.candidates(version)).inputs.sources) == 256
    assert fetched == []

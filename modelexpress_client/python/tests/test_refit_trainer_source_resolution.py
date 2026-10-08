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


def sources(
    monkeypatch, count, *, failed_replica=False
) -> tuple[trainer.TrainerSourceResolver, SimpleNamespace, list[str], int]:
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
        trainer_mesh_generation=1,
        base_version_id=None,
        layout_signature="layout",
        payload_format=WeightPayloadFormat.FULL_TENSOR,
    )
    return resolver, version, fetched, len(manifest)


@pytest.mark.parametrize("count", [1, 256])
def test_cold_shards_aggregate_fetch_counts(monkeypatch, count) -> None:
    resolver, version, fetched, size = sources(monkeypatch, count)
    recorder = RefitTimingRecorder(backend="rl_generator", version="v1", rank=0)
    with use_refit_timing(recorder):
        candidates = resolver.candidates(version)
        assert len(next(candidates).shards) == count
        stage = recorder.as_dict()["stages"]["source_preparation"]
        assert stage["count"] == count * 3 + 2
        assert stage["metadata"]["manifest_fetch_count"] == count
        assert stage["metadata"]["manifest_bytes"] == count * size
        assert "source_resolution_s" in stage["metadata"]
        before = dict(stage["metadata"])
        candidates.close()
        assert recorder.as_dict()["stages"]["source_preparation"]["metadata"] == before
    assert fetched == [
        f"worker-{index}"
        for index in sorted(range(count), key=lambda index: f"slot-{index}")
    ]


def test_mixed_shards_count_failed_fetches_and_successful_bytes(monkeypatch) -> None:
    resolver, version, fetched, size = sources(monkeypatch, 3, failed_replica=True)
    recorder = RefitTimingRecorder(backend="rl_generator", version="v1", rank=0)
    with use_refit_timing(recorder):
        candidates = resolver.candidates(version)
        assert len(next(candidates).shards) == 3
        candidates.close()
    metadata = recorder.as_dict()["stages"]["source_preparation"]["metadata"]
    assert metadata["manifest_fetch_count"] == 4
    assert metadata["manifest_fetch_bytes"] == 3 * size
    assert metadata["manifest_bytes"] == 3 * size
    assert "manifest_fetch_s" in metadata
    assert "manifest_hash_s" in metadata
    assert "manifest_fingerprint_s" in metadata
    assert fetched == ["a-failed", "worker-0", "worker-1", "worker-2"]


def test_warm_resolution_without_timing(monkeypatch) -> None:
    resolver, version, fetched, _ = sources(monkeypatch, 256)
    cached = next(resolver.candidates(version))
    resolver = trainer.TrainerSourceResolver(
        service=resolver._service, rpc_timeout_seconds=1, cached_source=lambda: cached
    )
    fetched.clear()
    assert len(next(resolver.candidates(version)).shards) == 256
    assert fetched == []


@pytest.mark.parametrize(
    "change",
    [None, "mesh_id", "generation", "debug", "digest", "layout_off", "plans_off"],
)
def test_mesh_cache_controls_publication_and_manifest_rpc_work(
    monkeypatch, change
) -> None:
    original, version, fetched, _ = sources(monkeypatch, 2)
    service = original._service()
    mesh_rpc, list_rpc = service.GetTrainerMesh, service.ListWeightVersionShards
    counts = {"mesh": 0, "list": 0}
    current = {"mesh_id": "mesh", "generation": 1}

    def get_mesh(request, **kwargs) -> refit_pb2.GetTrainerMeshResponse:
        counts["mesh"] += 1
        response = mesh_rpc(request, **kwargs)
        response.mesh.mesh_id = current["mesh_id"]
        response.mesh.generation = current["generation"]
        return response

    def list_shards(request, **kwargs) -> refit_pb2.ListWeightVersionShardsResponse:
        counts["list"] += 1
        return list_rpc(request, **kwargs)

    service.GetTrainerMesh, service.ListWeightVersionShards = get_mesh, list_shards
    settings = {
        "debug": ("MX_REFIT_DEBUG_VALIDATE_PLAN", "1"),
        "digest": ("MX_RESHARD_PUBLISH_DIGEST", "1"),
        "layout_off": ("MX_REFIT_CACHE_GENERATOR_LAYOUT", "0"),
        "plans_off": ("MX_REFIT_CACHE_PLAN", "0"),
    }
    if change in settings:
        monkeypatch.setenv(*settings[change])
    cached = [None]
    resolver = trainer.TrainerSourceResolver(
        service=lambda: service, rpc_timeout_seconds=1, cached_source=lambda: cached[0]
    )
    cached[0] = next(resolver.candidates(version))
    assert counts == {"mesh": 2, "list": 1}
    assert len(fetched) == 2
    version.version_id = "v2"
    if change == "mesh_id":
        current["mesh_id"] = version.trainer_mesh_id = "other"
    elif change == "generation":
        current["generation"] = version.trainer_mesh_generation = 2
    next(resolver.candidates(version))
    assert counts["list"] == (1 if change in (None, "layout_off") else 2)
    assert counts["mesh"] == (3 if change in (None, "layout_off") else 4)
    assert len(fetched) == (2 if change in (None, "layout_off") else 4)


def test_resumed_warm_candidate_discovers_fresh_replicas(monkeypatch) -> None:
    original, version, fetched, _ = sources(monkeypatch, 1)
    service = original._service()
    list_rpc = service.ListWeightVersionShards
    requests = []

    def list_shards(request, **kwargs) -> refit_pb2.ListWeightVersionShardsResponse:
        requests.append(request.version_id)
        return list_rpc(request, **kwargs)

    service.ListWeightVersionShards = list_shards
    cached = [None]
    resolver = trainer.TrainerSourceResolver(
        service=lambda: service, rpc_timeout_seconds=1, cached_source=lambda: cached[0]
    )
    cached[0] = next(resolver.candidates(version))
    version.version_id = "v2"
    candidates = resolver.candidates(version)
    next(candidates)
    assert requests == ["v1"]
    next(candidates)
    assert requests == ["v1", "v2"]
    assert len(fetched) == 2


@pytest.mark.parametrize("generation", [0, 2])
def test_version_generation_mismatch_fails_before_source_fetch_and_recovers(
    monkeypatch, generation
) -> None:
    resolver, version, fetched, _ = sources(monkeypatch, 1)
    version.trainer_mesh_generation = generation
    with pytest.raises(RuntimeError, match="generation"):
        next(resolver.candidates(version))
    assert fetched == []
    version.trainer_mesh_generation = 1
    assert len(next(resolver.candidates(version)).shards) == 1

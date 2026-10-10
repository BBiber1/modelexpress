# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import field, fields, make_dataclass, replace

import pytest
from modelexpress.refit.reshard.slice_plan import Shard
from modelexpress.refit.reshard.transfer_plan import SourceInfo
from modelexpress_rl.inference import nixl_staged_transfer as transfer
from modelexpress_rl.inference import _source_snapshot as snapshot

from tests.test_refit_warm_caches import _bounded_cache_inputs, _transfer_with_resolved_plan


def test_source_rows_are_isolated_from_mutable_parser_rows() -> None:
    original = _bounded_cache_inputs()["resolved"]
    frozen_sources = snapshot._freeze_sources(original.sources)

    assert frozen_sources is not None
    original.sources["weight"].shards[0].addr += 16
    original.sources["weight"].shards.reverse()
    assert frozen_sources["weight"].shards[0].addr == 100
    with pytest.raises(TypeError):
        frozen_sources["other"] = frozen_sources["weight"]
    with pytest.raises((AttributeError, TypeError)):
        frozen_sources["weight"].shards[0].addr = 999


def test_unrecognized_source_rows_keep_the_mutable_resolver_result() -> None:
    resolved = _bounded_cache_inputs()["resolved"]
    source = resolved.sources["weight"]
    source.extra = []

    assert snapshot._freeze_sources(resolved.sources) is None


@pytest.mark.parametrize(
    ("cls", "field", "replacement"),
    [
        (snapshot._ShardSnapshot, "addr", property(lambda self: 999)),
        (snapshot._TensorSnapshot, "global_shape", property(lambda self: (8,))),
    ],
)
def test_changed_frozen_row_accessors_are_not_retained(cls, field, replacement, monkeypatch) -> None:
    monkeypatch.setattr(cls, field, replacement)

    assert snapshot._freeze_sources(_bounded_cache_inputs()["resolved"].sources) is None


@pytest.mark.parametrize("cls", [snapshot._ShardSnapshot, snapshot._TensorSnapshot])
def test_frozen_rows_do_not_call_replaceable_constructors(monkeypatch, cls) -> None:
    def custom_new(*args, **kwargs) -> None:
        pytest.fail("private tuple constructor was invoked")

    monkeypatch.setattr(cls, "__new__", custom_new)
    frozen = snapshot._freeze_sources(_bounded_cache_inputs()["resolved"].sources)

    assert frozen is not None
    assert frozen["weight"].shards[0].addr == 100


def test_freeze_rechecks_row_classes_after_mid_copy_mutation(monkeypatch) -> None:
    original = snapshot._snapshot_classes_unchanged
    checked = False

    def mutate_after_initial_check() -> bool:
        nonlocal checked
        result = original()
        if not checked:
            checked = True
            monkeypatch.setattr(
                snapshot._ShardSnapshot, "addr", property(lambda self: 999)
            )
        return result

    monkeypatch.setattr(snapshot, "_snapshot_classes_unchanged", mutate_after_initial_check)

    assert snapshot._freeze_sources(_bounded_cache_inputs()["resolved"].sources) is None



@pytest.mark.parametrize(
    "field_name",
    [
        "shape",
        "offset",
        "source_shape",
        "address_bool",
        "dtype",
        "digest",
        "session",
        "source_extra",
        "shard_extra",
        "source_subclass",
        "shard_subclass",
        "source_dict_subclass",
    ],
)
def test_nonordinary_sources_keep_original_mutable_path(field_name) -> None:
    resolved = _bounded_cache_inputs()["resolved"]
    source = resolved.sources["weight"]
    shard = source.shards[0]
    if field_name == "shape":
        shard.shape = [2]
    elif field_name == "offset":
        shard.shard_offset = (True,)
    elif field_name == "source_shape":
        source.global_shape = [4]
    elif field_name == "address_bool":
        shard.addr = True
    elif field_name == "dtype":
        source.dtype = object()
    elif field_name == "digest":
        shard.digest = []
    elif field_name == "session":
        shard.session = object()
    elif field_name == "source_extra":
        source.extra = []
    elif field_name == "shard_extra":
        shard.extra = []
    elif field_name == "source_subclass":
        source.__class__ = type("CustomSource", (SourceInfo,), {})
    elif field_name == "shard_subclass":
        shard.__class__ = type("CustomShard", (Shard,), {})
    else:
        resolved = replace(
            resolved, sources=type("CustomDict", (dict,), {})(resolved.sources)
        )
    assert snapshot._freeze_sources(resolved.sources) is None


@pytest.mark.parametrize(
    "field_name", ["addr", "digest", "device_id", "agent_meta_b64"]
)
def test_changed_manifest_keeps_current_version_metadata(
    monkeypatch: pytest.MonkeyPatch, field_name: str,
) -> None:
    args = _bounded_cache_inputs()
    cache, _ = _transfer_with_resolved_plan(monkeypatch, args["manifests"])
    payload = json.loads(args["manifests"][0])
    values = {
        "addr": 900,
        "digest": "new-content",
        "device_id": 3,
        "agent_meta_b64": "bmV3",
    }
    target = (
        payload
        if field_name == "agent_meta_b64"
        else payload["tensors"][0]["shards"][0]
    )
    target[field_name] = values[field_name]
    manifests = [json.dumps(payload).encode(), args["manifests"][1]]
    second = cache._resolve_metadata(manifests, {})
    expected = transfer._resolve_sources(manifests)
    for item in fields(expected):
        if item.name != "sources":
            assert getattr(second, item.name) == getattr(expected, item.name)
    assert [shard.digest for shard in second.sources["weight"].shards] == [
        shard.digest for shard in expected.sources["weight"].shards
    ]


def test_metadata_preserves_extended_outer_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _bounded_cache_inputs()
    original = transfer._resolve_sources
    extra_maps = {
        "str_map": {"a": "DRAM"},
        "int_map": {"a": 1},
        "bytes_map": {"a": b"metadata"},
    }
    extended = make_dataclass(
        "ExtendedResolved",
        [(name, dict, field(default_factory=dict)) for name in extra_maps],
        bases=(transfer._ResolvedSources,),
        frozen=True,
    )
    parsed = []

    def resolve(*values, **kwargs) -> transfer._ResolvedSources:
        base = original(*values, **kwargs)
        value = extended(
            **{item.name: getattr(base, item.name) for item in fields(base)},
            **extra_maps,
        )
        parsed.append(value)
        return value

    monkeypatch.setattr(transfer, "_resolve_sources", resolve)
    cache, result = _transfer_with_resolved_plan(monkeypatch, args["manifests"])
    assert type(result) is extended
    for item in fields(result):
        if item.name != "sources":
            assert getattr(result, item.name) == getattr(parsed[0], item.name)
    reused = cache._resolve_metadata(args["manifests"], {})
    assert type(reused) is extended
    for item in fields(result):
        assert getattr(reused, item.name) == getattr(result, item.name)

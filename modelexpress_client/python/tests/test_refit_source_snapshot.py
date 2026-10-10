# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import json
import sys
from dataclasses import field, fields, make_dataclass, replace

import pytest
from modelexpress.refit.reshard.slice_plan import Shard
from modelexpress.refit.reshard.transfer_plan import SourceInfo
from modelexpress_rl.inference import _source_snapshot as snapshot
from modelexpress_rl.inference import nixl_staged_transfer as transfer

from tests.test_refit_warm_caches import (
    _bounded_cache_inputs,
    _transfer_with_resolved_plan,
)


def _freeze(resolved):
    token = snapshot._freeze_sources(resolved.sources)
    assert token is not None
    return replace(resolved, sources=token.sources), token


def test_source_snapshot_isolated_from_mutable_parser_rows(monkeypatch) -> None:
    args = _bounded_cache_inputs()
    original = args["resolved"]
    frozen, token = _freeze(original)
    expected = tuple(
        (name, snapshot._source_structure(source))
        for name, source in original.sources.items()
    )
    assert snapshot._snapshot_structure(frozen, token) == expected
    original.sources["weight"].shards[0].addr += 16
    original.sources["weight"].shards.reverse()
    assert snapshot._snapshot_structure(frozen, token) == expected
    assert frozen.sources["weight"].shards[0].addr == 100


@pytest.mark.parametrize("target", ["sources", "source", "shards", "shard", "shape"])
def test_source_snapshot_rejects_nested_writes(target):
    frozen, _token = _freeze(_bounded_cache_inputs()["resolved"])
    source = frozen.sources["weight"]
    mutate = {
        "sources": lambda: frozen.sources.__setitem__("new", source),
        "source": lambda: object.__setattr__(source, "elsize", 8),
        "shards": lambda: source.shards.append(source.shards[0]),
        "shard": lambda: object.__setattr__(source.shards[0], "addr", 999),
        "shape": lambda: source.global_shape.__setitem__(0, 7),
    }[target]
    with pytest.raises((AttributeError, TypeError)):
        mutate()


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
def test_nonordinary_sources_keep_original_mutable_path(field_name):
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
    assert snapshot._snapshot_structure(resolved, None) is None


@pytest.mark.parametrize("field_name", ["addr", "session", "shape"])
def test_changed_snapshot_accessor_rebuilds_current_source_fields(
    monkeypatch, field_name
) -> None:
    args = _bounded_cache_inputs()
    frozen, token = _freeze(args["resolved"])
    args.update(resolved=frozen, source_snapshot=token)
    replacement = {"addr": 900, "session": "new-session", "shape": (1,)}[field_name]
    monkeypatch.setattr(
        snapshot._ShardSnapshot, field_name, property(lambda self: replacement)
    )
    assert snapshot._snapshot_structure(frozen, token) is None
    assert (
        snapshot._source_structure(frozen.sources["weight"])
        != tuple.__getitem__(token, 1)[0][1]
    )


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
    second, token = cache._resolve_metadata(manifests, {})
    expected = transfer._resolve_sources(manifests)
    for item in fields(expected):
        if item.name != "sources":
            assert getattr(second, item.name) == getattr(expected, item.name)
    assert [shard.digest for shard in second.sources["weight"].shards] == [
        shard.digest for shard in expected.sources["weight"].shards
    ]
    assert snapshot._snapshot_structure(second, token) == tuple(
        (name, snapshot._source_structure(source))
        for name, source in expected.sources.items()
    )


@pytest.mark.parametrize("which", ["agents", "devices", "metadata"])
def test_outer_metadata_remains_fresh_and_mutable(which) -> None:
    args = _bounded_cache_inputs()
    original = args["resolved"]
    resolved, token = _freeze(original)
    args.update(resolved=resolved, source_snapshot=token)
    if which == "agents":
        resolved.session_to_agent["a"] = "changed"
    elif which == "devices":
        resolved.session_to_device["a"] = 7
    else:
        resolved.agent_metadata["a"] = b"changed"
    assert getattr(original, {
        "agents": "session_to_agent",
        "devices": "session_to_device",
        "metadata": "agent_metadata",
    }[which])["a"] == (7 if which == "devices" else b"changed" if which == "metadata" else "changed")


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
    reused, _ = cache._resolve_metadata(args["manifests"], {})
    assert type(reused) is extended
    for item in fields(result):
        assert getattr(reused, item.name) == getattr(result, item.name)


def test_snapshot_token_requires_current_table_identity():
    original = _bounded_cache_inputs()["resolved"]
    first, token = _freeze(original)
    second, other = _freeze(original)
    assert snapshot._snapshot_structure(first, token) is not None
    assert snapshot._snapshot_structure(first, other) is None
    assert snapshot._snapshot_structure(second, token) is None
    assert snapshot._snapshot_structure(original, token) is None


def test_metadata_freeze_failure_propagates_and_later_resolution_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _bounded_cache_inputs()
    cache, first = _transfer_with_resolved_plan(monkeypatch, args["manifests"])
    changed = [args["manifests"][0] + b" ", args["manifests"][1]]
    original = transfer._freeze_sources
    monkeypatch.setattr(
        transfer, "_freeze_sources", lambda _: (_ for _ in ()).throw(MemoryError())
    )
    with pytest.raises(MemoryError):
        cache._resolve_metadata(changed, {})
    monkeypatch.setattr(transfer, "_freeze_sources", original)
    resolved, token = cache._resolve_metadata(changed, {})
    assert token is not None
    expected = transfer._resolve_sources(changed)
    expected_structure = tuple(
        (name, snapshot._source_structure(source))
        for name, source in expected.sources.items()
    )
    assert snapshot._snapshot_structure(resolved, token) == expected_structure
    resolved, token = cache._resolve_metadata(args["manifests"], {})
    assert token is not None
    assert snapshot._snapshot_structure(resolved, token) == expected_structure
    for item in fields(first):
        assert getattr(resolved, item.name) == getattr(first, item.name)


@pytest.mark.parametrize(
    "cls", [snapshot._ShardSnapshot, snapshot._TensorSnapshot, snapshot._SourceSnapshot]
)
def test_private_snapshot_construction_cannot_inject_mutable_fields(monkeypatch, cls):
    def custom_new(*args, **kwargs):
        pytest.fail("private tuple construction invoked a replaceable constructor")

    monkeypatch.setattr(cls, "__new__", custom_new)
    frozen, token = _freeze(_bounded_cache_inputs()["resolved"])
    assert snapshot._snapshot_structure(frozen, token) is not None


def test_warm_class_checks_do_not_return_a_key_stale_after_gc_callback():
    frozen, token = _freeze(_bounded_cache_inputs()["resolved"])
    original = snapshot._ShardSnapshot.addr
    changed = False

    def change_already_checked_class(phase, info):
        nonlocal changed
        if phase != "start" or changed:
            return
        frame = sys._getframe(1)
        while frame:
            if (
                frame.f_code is snapshot._snapshot_classes_unchanged.__code__
                and frame.f_locals.get("cls") is snapshot._TensorSnapshot
            ):
                snapshot._ShardSnapshot.addr = property(lambda self: 999)
                changed = True
                return
            frame = frame.f_back

    previous = gc.get_threshold()
    gc.collect()
    gc.callbacks.append(change_already_checked_class)
    gc.set_threshold(1, 1, 1)
    try:
        key = snapshot._snapshot_structure(frozen, token)
        fresh = tuple(
            (name, snapshot._source_structure(source))
            for name, source in frozen.sources.items()
        )
        assert key is None or key == fresh
    finally:
        gc.set_threshold(*previous)
        gc.callbacks.remove(change_already_checked_class)
        snapshot._ShardSnapshot.addr = original

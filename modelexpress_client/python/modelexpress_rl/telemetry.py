# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-facing refit scopes and cross-process trace ownership."""

from __future__ import annotations

import contextlib
import contextvars
import time
from collections.abc import Iterator, Mapping
from typing import Any

from modelexpress import telemetry

_active_refit: contextvars.ContextVar[RefitTrace | None] = contextvars.ContextVar(
    "mx_refit_owner", default=None
)


class RefitTrace:
    """Own one role and its optional root/group without leaving context attached."""

    def __init__(
        self,
        role: str,
        *,
        rank: int = 0,
        step: int | None = None,
        staging_mode: str | None = None,
    ) -> None:
        self._name = f"mx.refit.{role}"
        self._attributes: dict[str, Any] = {
            **telemetry.refit_metadata(step=step, staging_mode=staging_mode),
            "role": role,
            "rank": rank,
            "refit.aggregate": True,
        }
        self._started = time.time_ns() if telemetry.enabled() else None
        self._parent: Mapping[str, str] = {}
        self._role: telemetry.RefitCycle | None = None
        self._root: telemetry.RefitCycle | None = None
        self._group: telemetry.RefitCycle | None = None
        self._pending: list[tuple[str, int, int, Mapping[str, Any] | None]] = []
        self._finished = False

    @classmethod
    def trainer(
        cls,
        *,
        step: int,
        rank: int,
        trainers: int,
        generators: int,
        staging_mode: str,
    ) -> RefitTrace:
        telemetry.configure("prime-rl-trainer")
        trace = cls("trainer", rank=rank, step=step, staging_mode=staging_mode)
        if rank == 0:
            attributes = {
                k: v for k, v in trace._attributes.items() if k not in ("role", "rank")
            }
            trace._root = telemetry.RefitCycle(
                {
                    **attributes,
                    "refit.root": True,
                    "refit.expected_trainers": trainers,
                    "refit.expected_generators": generators,
                },
                start_time=trace._started,
            )
            trace._group = telemetry.RefitCycle(
                {**attributes, "role": "trainer"},
                name="mx.refit.trainers",
                parent=trace.context()["root"],
                start_time=trace._started,
            )
            trace.bind(trace.context()["trainers"])
        return trace

    @classmethod
    def orchestrator(cls, *, step: int, staging_mode: str) -> RefitTrace:
        telemetry.configure("prime-rl-orchestrator")
        return cls("orchestrator", step=step, staging_mode=staging_mode)

    @classmethod
    def generator(
        cls,
        *,
        version_uid: str,
        rank: int,
        parent: Mapping[str, str] | None = None,
    ) -> RefitTrace:
        telemetry.configure("prime-rl-inference")
        trace = cls("generator", rank=rank)
        trace.bind(parent or {}, version_uid=version_uid)
        return trace

    def context(self) -> dict[str, dict[str, str]]:
        """Export root and trainer-group parents for the existing rank broadcast."""
        carriers: dict[str, dict[str, str]] = {"root": {}, "trainers": {}}
        for key, cycle in (("root", self._root), ("trainers", self._group)):
            if cycle is not None:
                cycle.inject(carriers[key])
        return carriers

    def set_version(self, version_uid: str) -> None:
        attributes = {"version_uid": version_uid, "refit.id": version_uid}
        self._attributes.update(attributes)
        for cycle in (self._root, self._group, self._role):
            if cycle is not None:
                cycle.set_attributes(attributes)

    def bind(
        self, parent: Mapping[str, str], *, version_uid: str | None = None
    ) -> None:
        """Attach a discovered parent and emit operations measured before it arrived."""
        if self._finished:
            raise RuntimeError("refit trace is finished")
        if version_uid is not None:
            self.set_version(version_uid)
        if self._role is not None:
            return
        self._parent = dict(parent)
        with telemetry.extracted(self._parent):
            self._role = telemetry.RefitCycle(
                self._attributes,
                name=self._name,
                parent=self._parent,
                start_time=self._started,
            )
        with self.active():
            for name, start, end, attributes in self._pending:
                telemetry.completed_span(name, start, end, attributes)
        self._pending.clear()

    @contextlib.contextmanager
    def active(self) -> Iterator[RefitTrace]:
        """Activate the role for MX calls; end the owned spans if work fails."""
        if _active_refit.get() is self:
            yield self
            return
        token = _active_refit.set(self)
        try:
            if self._role is None:
                with telemetry.untraced():
                    yield self
            else:
                with (
                    telemetry.extracted(self._parent),
                    self._role.active(),
                    telemetry.refit_attributes(
                        self._attributes,
                        role=self._attributes["role"],
                        rank=self._attributes["rank"],
                    ),
                ):
                    yield self
        except BaseException as error:
            self.finish(error)
            raise
        finally:
            _active_refit.reset(token)

    @contextlib.contextmanager
    def span(
        self, name: str, attributes: Mapping[str, Any] | None = None
    ) -> Iterator[None]:
        """Measure a framework operation, deferring export until bind if necessary."""
        name = f"mx.refit.{name}"
        with self.active():
            if self._role is not None:
                with telemetry.span(name, attributes):
                    yield
            elif telemetry.enabled():
                start = time.time_ns()
                yield
                self._pending.append((name, start, time.time_ns(), attributes))
            else:
                yield

    @contextlib.contextmanager
    def generators(self) -> Iterator[dict[str, str]]:
        """Group inference workers under the root without changing the caller's role."""
        with self.active():
            with telemetry.refit_attributes(self._attributes, role="generator"):
                group = telemetry.RefitCycle(
                    {"role": "generator"},
                    name="mx.refit.generators",
                    parent=self._parent,
                )
            carrier: dict[str, str] = {}
            group.inject(carrier)
            try:
                yield carrier
            except BaseException as error:
                group.finish(error)
                raise
            else:
                group.finish()

    def finish(self, error: BaseException | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        self._pending.clear()
        for cycle in (self._role, self._group, self._root):
            if cycle is not None:
                cycle.finish(error)

    def __enter__(self) -> RefitTrace:  # noqa: PYI034 - Python 3.10 support.
        return self

    def __exit__(self, exc_type, error, traceback) -> None:
        self.finish(error)

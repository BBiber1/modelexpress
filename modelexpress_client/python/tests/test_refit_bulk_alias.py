# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python alias validation preserves live ownership and callback ordering."""

import pytest
import torch
from modelexpress_rl.inference.engines.vllm import installer as api


def _fixture():
    model = torch.nn.Module()
    model.branch = torch.nn.Linear(2, 2)
    model.alias = model.branch
    installer = api._VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    return model, installer


@pytest.mark.parametrize("mutate_before_capture", [False, True])
def test_python_owner_check_observes_replaced_getattr(
    monkeypatch, mutate_before_capture
):
    model, installer = _fixture()
    aliases = None if mutate_before_capture else installer._parameter_aliases(model)

    def changed_getattr(self, name):
        raise RuntimeError("changed module lookup")

    monkeypatch.setattr(torch.nn.Module, "__getattr__", changed_getattr)
    if aliases is None:
        aliases = installer._parameter_aliases(model)
    with pytest.raises(RuntimeError, match="changed module lookup"):
        installer._validate_alias_owners(aliases)


def test_python_owner_check_rejects_replaced_module():
    model, installer = _fixture()
    aliases = installer._parameter_aliases(model)
    installer._validate_alias_owners(aliases)
    model.branch = torch.nn.Linear(2, 2)
    with pytest.raises(api.IncompleteRefit, match="replaced during refit"):
        installer._validate_alias_owners(aliases)


def test_python_writer_check_reads_replaced_registries_and_instance_hooks():
    model, installer = _fixture()
    aliases = installer._parameter_aliases(model)
    assert api._standard_parameter_writes(aliases)
    model.branch._parameters = dict(model.branch._parameters)
    model.branch.weight = torch.nn.Parameter(torch.ones(2, 2))
    assert api._standard_parameter_writes(aliases)
    model.branch.register_parameter = lambda name, value: None
    assert not api._standard_parameter_writes(aliases)


def test_python_writer_check_does_not_repeat_custom_hook_truth(monkeypatch):
    model, installer = _fixture()
    aliases = installer._parameter_aliases(model)
    calls = []

    class Hooks(dict):
        def __bool__(self):
            calls.append("bool")
            return len(calls) > 1

    monkeypatch.setattr(
        torch.nn.modules.module, "_global_parameter_registration_hooks", Hooks()
    )
    assert api._standard_parameter_writes(aliases)
    assert calls == ["bool"]

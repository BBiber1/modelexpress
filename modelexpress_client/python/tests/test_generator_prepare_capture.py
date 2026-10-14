# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from modelexpress.adapter import StrategyRecoveryError
from modelexpress.refit.reshard.types import CaptureResult
from modelexpress_rl.inference.engines.vllm import installer as module
from tests.test_refit_vllm_installer import _install_fake_vllm


@pytest.mark.parametrize("failure", ["initialize", "restore", "reset", "capture"])
def test_capture_failure_restores_layers_and_marks_unsafe_failures(monkeypatch, failure):
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False))
    originals = [layer.weight for layer in model]
    restored = []

    class Info:
        def __init__(self, parameter):
            self.kernel_tensors = ({"weight": parameter}, {})

        def reset(self):
            if failure == "reset" and self.kernel_tensors[0]["weight"] is originals[0]:
                raise RuntimeError("reset failed")
            self.kernel_tensors = None

    def initialize(target):
        for layer in target:
            layerwise.LAYERWISE_INFO[layer] = Info(layer.weight)
            layer.weight = nn.Parameter(torch.empty_like(layer.weight, device="meta"))
        if failure == "initialize":
            raise RuntimeError("initialize failed")

    _install_fake_vllm(monkeypatch, initialize)
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    layerwise._get_original_loader = lambda parameter: None

    def restore(layer, info):
        if failure == "restore" and layer is model[0]:
            raise RuntimeError("restore failed")
        layer.weight = info.kernel_tensors[0]["weight"]
        restored.append(layer)

    layerwise._place_kernel_tensors = restore
    weight_utils = ModuleType("vllm.model_executor.model_loader.weight_utils")
    weight_utils.default_weight_loader = lambda parameter, weight: None
    monkeypatch.setitem(sys.modules, weight_utils.__name__, weight_utils)
    capture = Mock(return_value=CaptureResult(copies=[]))
    if failure == "capture":
        capture.side_effect = RuntimeError("capture failed")
    monkeypatch.setattr(module, "capture_weights", capture)
    installer = module._VllmInstaller(
        model=model, vllm_config=object(), model_config=object(), device=torch.device("cpu"),
    )
    with pytest.raises(RuntimeError) as raised:
        installer.capture([("weight", torch.float32, (2, 2))])
    assert isinstance(raised.value, StrategyRecoveryError) == (failure != "capture")
    assert model[1] in restored
    assert model[1].weight is originals[1]
    if failure != "restore":
        assert model[0].weight is originals[0]

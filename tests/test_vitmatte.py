import importlib.util
import sys
import types

import numpy as np
import pytest
import torch

from ai_segmenter.models.vitmatte import ViTMatteModel


class AttrOutput:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def make_instance(**overrides):
    instance = ViTMatteModel.__new__(ViTMatteModel)
    instance.torch = torch
    instance.device = torch.device("cpu")
    instance.device_label = "CPU"
    instance.input_size = 8
    instance.use_autocast = False
    instance.autocast_dtype = None
    instance.tensorrt_enabled = False
    for key, value in overrides.items():
        setattr(instance, key, value)
    return instance


def test_missing_required_module_raises_helpful_error(monkeypatch):
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "timm":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(RuntimeError, match="timm"):
        ViTMatteModel()


def test_heuristic_trimap_marks_bright_as_foreground_and_dark_as_background():
    frame = np.full((30, 30, 3), 128, dtype=np.uint8)
    frame[0:10, 0:10] = 255
    frame[20:30, 20:30] = 0

    instance = ViTMatteModel.__new__(ViTMatteModel)
    trimap = instance._heuristic_trimap(frame)

    assert trimap.shape == (30, 30)
    assert trimap[5, 5] == 255
    assert trimap[25, 25] == 0
    assert trimap[15, 0] == 128


def test_prepare_inputs_resizes_image_and_trimap_to_input_size():
    instance = make_instance(input_size=16)
    rgb_frame = np.random.randint(0, 255, (40, 60, 3), dtype=np.uint8)

    image, trimap = instance._prepare_inputs(rgb_frame)

    assert image.size == (16, 16)
    assert trimap.size == (16, 16)
    assert trimap.mode == "L"


def test_forward_extracts_alpha_from_alphas_attribute():
    alpha_tensor = torch.full((1, 1, 8, 8), 0.75)
    instance = make_instance(
        processor=lambda images, trimaps, return_tensors="pt": {"pixel_values": torch.zeros((1, 4, 8, 8))},
        model=lambda **kwargs: AttrOutput(alphas=alpha_tensor),
    )

    result = instance._forward(image=None, trimap=None)

    assert torch.equal(result, alpha_tensor)


def test_forward_extracts_alpha_from_alpha_attribute_when_alphas_missing():
    alpha_tensor = torch.full((1, 1, 8, 8), 0.25)
    instance = make_instance(
        processor=lambda images, trimaps, return_tensors="pt": {"pixel_values": torch.zeros((1, 4, 8, 8))},
        model=lambda **kwargs: AttrOutput(alpha=alpha_tensor),
    )

    result = instance._forward(image=None, trimap=None)

    assert torch.equal(result, alpha_tensor)


def test_forward_extracts_alpha_from_tuple_output():
    alpha_tensor = torch.zeros((1, 1, 8, 8))
    instance = make_instance(
        processor=lambda images, trimaps, return_tensors="pt": {"pixel_values": torch.zeros((1, 4, 8, 8))},
        model=lambda **kwargs: (alpha_tensor, "extra"),
    )

    result = instance._forward(image=None, trimap=None)

    assert torch.equal(result, alpha_tensor)


def test_forward_raises_when_model_returns_no_alpha():
    instance = make_instance(
        processor=lambda images, trimaps, return_tensors="pt": {"pixel_values": torch.zeros((1, 4, 8, 8))},
        model=lambda **kwargs: AttrOutput(),
    )

    with pytest.raises(RuntimeError, match="Alpha"):
        instance._forward(image=None, trimap=None)


def test_predict_mask_interpolates_and_clips_alpha_to_uint8(monkeypatch):
    instance = make_instance()
    rgb_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    small_alpha = torch.tensor([[[[2.0, -1.0], [0.5, 0.5]]]])

    monkeypatch.setattr(instance, "_prepare_inputs", lambda frame: (None, None))
    monkeypatch.setattr(instance, "_forward", lambda image, trimap: small_alpha)

    mask = instance.predict_mask(rgb_frame)

    assert mask.shape == (4, 4)
    assert mask.dtype == np.uint8
    assert mask.max() == 255
    assert mask.min() == 0


def test_predict_mask_skips_interpolation_when_shapes_already_match(monkeypatch):
    instance = make_instance()
    rgb_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    alpha = torch.full((1, 1, 4, 4), 0.5)

    monkeypatch.setattr(instance, "_prepare_inputs", lambda frame: (None, None))
    monkeypatch.setattr(instance, "_forward", lambda image, trimap: alpha)

    def fail_interpolate(*args, **kwargs):
        raise AssertionError("interpolate should not run when shapes already match")

    monkeypatch.setattr(torch.nn.functional, "interpolate", fail_interpolate)

    mask = instance.predict_mask(rgb_frame)

    assert mask.shape == (4, 4)
    assert (mask == 127).all()


def test_compile_tensorrt_raises_when_device_is_not_cuda():
    instance = make_instance(device_label="CPU")

    with pytest.raises(RuntimeError, match="CUDA"):
        instance._compile_tensorrt()


def test_compile_tensorrt_uses_single_four_channel_pixel_values_input(monkeypatch):
    captured = {}

    class FakeInput:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

    def fake_compile(model, ir, inputs, **kwargs):
        captured["inputs"] = inputs
        return model

    fake_module = types.SimpleNamespace(compile=fake_compile, Input=FakeInput)
    monkeypatch.setitem(sys.modules, "torch_tensorrt", fake_module)
    monkeypatch.setattr("ai_segmenter.models.vitmatte.prepare_tensorrt_import", lambda: None)

    calls = []

    def fake_model(pixel_values):
        calls.append(pixel_values)
        return AttrOutput(alphas=torch.zeros((1, 1, 8, 8)))

    instance = make_instance(device_label="CUDA", model=fake_model)

    instance._compile_tensorrt()

    assert instance.tensorrt_enabled is True
    assert len(captured["inputs"]) == 1
    assert captured["inputs"][0].shape == (1, 4, instance.input_size, instance.input_size)
    assert len(calls) == 1
    assert calls[0].shape == (1, 4, instance.input_size, instance.input_size)


def test_compile_tensorrt_falls_back_to_pytorch_on_failure(monkeypatch):
    def fake_compile(model, ir, inputs, **kwargs):
        raise RuntimeError("boom")

    fake_module = types.SimpleNamespace(compile=fake_compile, Input=lambda shape, dtype: None)
    monkeypatch.setitem(sys.modules, "torch_tensorrt", fake_module)
    monkeypatch.setattr("ai_segmenter.models.vitmatte.prepare_tensorrt_import", lambda: None)

    original_model = object()
    instance = make_instance(device_label="CUDA", model=original_model)

    instance._compile_tensorrt()

    assert instance.tensorrt_enabled is False
    assert instance.model is original_model
    assert "boom" in instance.device_hint

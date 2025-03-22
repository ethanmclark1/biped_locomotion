import os
import sys
import glob
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def item(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu()


def save_onnx(
    path: str, model: nn.Module, input_size: int, input_names: list, output_names: list
) -> None:
    from_device(model)
    torch.onnx.export(
        model,  # model being run
        torch.randn(1, input_size),  # model input (or a tuple for multiple inputs)
        path,  # where to save the model (can be a file or file-like object)
        training=False,
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=9,  # the ONNX version to export the model to
        do_constant_folding=False,  # whether to execute constant folding for optimization
        input_names=input_names,  # the model's input names
        output_names=output_names,  # the model's output names
    )
    to_device(model)


def to_device(x: nn.Module) -> nn.Module:
    return x.cuda() if torch.cuda.is_available() else x


def from_device(x: nn.Module) -> nn.Module:
    return x.cpu() if torch.cuda.is_available() else x


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.data.cpu().numpy()


def get_data_paths(data_path: str, version_no: str) -> list:
    sequence_dirs = glob.glob(os.path.join(data_path, "sequence*"))
    data_paths = [os.path.join(seq_dir, version_no) for seq_dir in sequence_dirs]

    return data_paths


def normalize(x: torch.Tensor, norm: np.ndarray) -> torch.Tensor:
    mean = norm[0]
    std = norm[1]
    return (x - mean) / std


def renormalize(x: torch.Tensor, norm: np.ndarray) -> torch.Tensor:
    mean = norm[0]
    std = norm[1]
    return (x * std) + mean

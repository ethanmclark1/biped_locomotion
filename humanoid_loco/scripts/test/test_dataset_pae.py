import yaml
import numpy as np
import matplotlib.pyplot as plt

import humanoid_loco.scripts.utils.helper as helper

from torch.utils.data import DataLoader
from humanoid_loco.scripts.pae.dataset import DatasetPAE


def plot_joint_vel_history(
    dataloader: DataLoader, n_joints: int, frames: int, window_size: int
) -> None:

    joint_vel_history = []

    for i, data in enumerate(data_loader):
        if i == window_size:
            break

        joint_vel = data
        joint_vel = helper.item(joint_vel).numpy()
        joint_vel_history.append(joint_vel)

    joint_vel_history = np.array(joint_vel_history)
    joint_vel_history = joint_vel_history.reshape(n_joints, -1)

    fig, axes = plt.subplots(6, 4, figsize=(20, 15))
    axes = axes.flatten()
    for j in range(n_joints):
        axes[j].plot(joint_vel_history[j, :], alpha=0.5, linewidth=1)
        axes[j].set_title(f"Joint {j} Velocity")
        axes[j].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_joint_state_history(
    data_loader: DataLoader, n_joints: int, frames: int, window_size: int
) -> None:

    joint_pos_history = []
    joint_vel_history = []

    centered_frame = frames // 2

    for i, data in enumerate(data_loader):
        if i == window_size:
            break

        joint_pos = data[:, :n_joints, centered_frame]
        joint_vel = data[:, n_joints:, centered_frame]

        joint_pos = helper.item(joint_pos).numpy()
        joint_vel = helper.item(joint_vel).numpy()

        joint_pos_history.append(joint_pos)
        joint_vel_history.append(joint_vel)

    joint_pos_history = np.array(joint_pos_history)
    joint_vel_history = np.array(joint_vel_history)

    fig, axes = plt.subplots(6, 4, figsize=(20, 15))
    axes = axes.flatten()
    for j in range(n_joints):
        axes[j].plot(joint_pos_history[..., j], alpha=0.5, linewidth=1)
        axes[j].set_title(f"Joint {j} Position")
        axes[j].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(6, 4, figsize=(20, 15))
    axes = axes.flatten()
    for j in range(n_joints):
        axes[j].plot(joint_vel_history[..., j], alpha=0.5, linewidth=1)
        axes[j].set_title(f"Joint {j} Velocity")
        axes[j].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    config_path = "humanoid_loco/scripts/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    seed = config["seed"]
    n_joints = config["n_joints"]
    frames = int(config["fps"] * config["window"]) + 1
    validation_ratio = config["validation_ratio"]
    datapath = config["datapath"]
    full_joint_state = config["PAE"]["full_joint_state"]

    dataset = DatasetPAE(
        seed, n_joints, frames, validation_ratio, datapath, full_joint_state
    )
    dataset.set_mode("eval")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    window_size = 1

    if full_joint_state:
        plot_joint_state_history(data_loader, n_joints, frames, window_size)
    else:
        plot_joint_vel_history(data_loader, n_joints, frames, window_size)

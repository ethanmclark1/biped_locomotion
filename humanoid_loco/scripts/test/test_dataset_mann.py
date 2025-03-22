import yaml
import numpy as np
import matplotlib.pyplot as plt

import humanoid_loco.scripts.utils.helper as helper

from torch.utils.data import DataLoader
from humanoid_loco.scripts.mann.dataset import (
    DatasetMANN,
    INPUT_STRUCTURE,
    TARGET_STRUCTURE,
)


def validate_dataset(dataset: DatasetMANN) -> None:
    input, output = dataset[0]

    assert input.shape[0] == list(INPUT_STRUCTURE.values())[-1].stop
    assert output.shape[0] == list(TARGET_STRUCTURE.values())[-1].stop


def plot_cmd_vels(data_loader: DataLoader, first_frame: int, window_size: int) -> None:
    cmd_vels_xy_vel = []
    cmd_vels_yaw_history = []

    for i, (input, _) in enumerate(data_loader):
        if i < first_frame:
            continue
        elif i == first_frame + window_size:
            break

        cmd_xy_vel = (
            helper.item(input[..., INPUT_STRUCTURE["cmd_vels_linear_xy"]])
            .squeeze(0)
            .numpy()
        )
        cmd_vels_yaw = (
            helper.item(input[..., INPUT_STRUCTURE["cmd_vels_yaw"]]).squeeze(0).numpy()
        )

        cmd_xy_vel = cmd_xy_vel.reshape(2, -1)
        cmd_vels_yaw = cmd_vels_yaw.reshape(-1)

        cmd_vels_xy_vel.append(cmd_xy_vel)
        cmd_vels_yaw_history.append(cmd_vels_yaw)

    cmd_vels_xy_vel = np.array(cmd_vels_xy_vel)
    cmd_vels_yaw_history = np.array(cmd_vels_yaw_history)

    num_frames = cmd_vels_xy_vel.shape[2]
    time_range = np.arange(first_frame, first_frame + window_size)
    time_labels = list(range(-6, 7))

    plt.figure(figsize=(24, 8))
    for i, value in enumerate(["X", "Y"]):
        plt.subplot(1, 3, i + 1)
        for j in range(num_frames):
            shifted_time = time_range + j
            plt.plot(
                shifted_time,
                cmd_vels_xy_vel[:, i, j],
                alpha=0.3 if j != 6 else 1.0,
                linewidth=1 if j != 6 else 2,
                label=f"t{'-' if j==0 else '+' if j==12 else ''}{abs(time_labels[j])}"
                if j in [0, 6, 12]
                else None,
            )
        plt.title(f"Command {value} Velocity")
        plt.xlabel("Frame")
        plt.ylabel(f"{value} Velocity")
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.subplot(1, 3, 3)
    for j in range(num_frames):
        shifted_time = time_range + j
        plt.plot(
            shifted_time,
            cmd_vels_yaw_history[:, j],
            alpha=0.3 if j != 6 else 1.0,
            linewidth=1 if j != 6 else 2,
            label=f"t{'-' if j==0 else '+' if j==12 else ''}{abs(time_labels[j])}"
            if j in [0, 6, 12]
            else None,
        )
    plt.title("Command Yaw Velocity")
    plt.xlabel("Frame")
    plt.ylabel("Yaw Velocity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)


def plot_character(data_loader: DataLoader, first_frame: int, window_size: int) -> None:
    root_projected_xy_gravity_history = []
    robot_oriented_xy_vel_history = []
    robot_oriented_yaw_vel_history = []

    joint_pos_history = []
    joint_vel_history = []

    foot_contacts_history = []

    for i, (_, target) in enumerate(data_loader):
        if i < first_frame:
            continue
        elif i == first_frame + window_size:
            break

        root_projected_xy_gravity = (
            helper.item(target[..., TARGET_STRUCTURE["robot_projected_gravity"]])
            .squeeze(0)
            .numpy()
        )
        robot_oriented_xy_vel = (
            helper.item(target[..., TARGET_STRUCTURE["robot_oriented_xy_vel"]])
            .squeeze(0)
            .numpy()
        )
        robot_oriented_yaw_vel = (
            helper.item(target[..., TARGET_STRUCTURE["robot_oriented_yaw_vel"]])
            .squeeze(0)
            .numpy()
        )

        joint_pos = (
            helper.item(target[..., TARGET_STRUCTURE["joint_positions"]])
            .squeeze(0)
            .numpy()
        )
        joint_vel = (
            helper.item(target[..., TARGET_STRUCTURE["joint_velocities"]])
            .squeeze(0)
            .numpy()
        )
        foot_contacts = (
            helper.item(target[..., TARGET_STRUCTURE["foot_contacts"]])
            .squeeze(0)
            .numpy()
        )

        root_projected_xy_gravity_history.append(root_projected_xy_gravity)
        robot_oriented_xy_vel_history.append(robot_oriented_xy_vel)
        robot_oriented_yaw_vel_history.append(robot_oriented_yaw_vel)

        joint_pos_history.append(joint_pos)
        joint_vel_history.append(joint_vel)

        foot_contacts_history.append(foot_contacts)

    root_projected_xy_gravity_history = np.array(root_projected_xy_gravity_history)
    robot_oriented_xy_vel_history = np.array(robot_oriented_xy_vel_history)
    robot_oriented_yaw_vel_history = np.array(robot_oriented_yaw_vel_history)

    joint_pos_history = np.array(joint_pos_history)
    joint_vel_history = np.array(joint_vel_history)

    foot_contacts_history = np.array(foot_contacts_history)

    plt.figure(figsize=(24, 12))
    values = "XYZ"
    for i in range(root_projected_xy_gravity_history.shape[1]):
        plt.subplot(1, root_projected_xy_gravity_history.shape[1], i + 1)
        plt.plot(
            np.arange(first_frame, first_frame + window_size),
            root_projected_xy_gravity_history[:, i],
            alpha=0.5,
            linewidth=1,
        )
        plt.title(f"Root Projected Gravity {values[i]}")
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)

    plt.figure(figsize=(24, 12))
    for i, value in enumerate(["X", "Y"]):
        plt.subplot(1, 3, i + 1)
        plt.plot(
            np.arange(first_frame, first_frame + window_size),
            robot_oriented_xy_vel_history[:, i],
            alpha=0.5,
            linewidth=1,
        )
        plt.title(f"Root Velocity {value}")
        plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(
        np.arange(first_frame, first_frame + window_size),
        robot_oriented_yaw_vel_history,
        alpha=0.5,
        linewidth=1,
    )
    plt.title("Root Yaw Velocity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)

    plt.figure(figsize=(24, 12))
    for i in range(23):
        plt.subplot(6, 4, i + 1)
        plt.plot(
            np.arange(first_frame, first_frame + window_size),
            joint_pos_history[:, i],
            alpha=0.5,
            linewidth=1,
        )
        plt.title(f"Joint {i} Position")
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)

    plt.figure(figsize=(24, 12))
    for i in range(23):
        plt.subplot(6, 4, i + 1)
        plt.plot(
            np.arange(first_frame, first_frame + window_size),
            joint_vel_history[:, i],
            alpha=0.5,
            linewidth=1,
        )
        plt.title(f"Joint {i} Velocity")
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)

    plt.figure(figsize=(24, 12))
    for i, value in enumerate(["Left", "Right"]):
        plt.subplot(1, 2, i + 1)
        plt.plot(
            np.arange(first_frame, first_frame + window_size),
            foot_contacts_history[:, i],
            alpha=0.5,
            linewidth=1,
        )
        plt.title(f"{value} Foot Contact")
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)


def plot_phases(dataloader: DataLoader, sampled_frames: int, window_size: int) -> None:
    sin_history = []
    cos_history = []

    current_frame = sampled_frames // 2

    for i, (input, _) in enumerate(dataloader):
        if i < current_frame:
            continue
        elif i == current_frame + window_size:
            break

        input_phase = input[:, INPUT_STRUCTURE["phase_space"]].reshape(
            -1, phase_channels, sampled_frames, 2
        )

        current_input_phase = input_phase[:, :, current_frame, :]

        input_sin = helper.item(current_input_phase[..., 0]).squeeze(0).numpy()
        input_cos = helper.item(current_input_phase[..., 1]).squeeze(0).numpy()

        sin_history.append(input_sin)
        cos_history.append(input_cos)

    sin_history = np.array(sin_history)
    cos_history = np.array(cos_history)

    plt.figure(figsize=(24, 12))
    for j in range(phase_channels):
        plt.subplot(2, 4, j + 1)
        plt.plot(
            np.arange(current_frame, current_frame + window_size),
            sin_history[:, j],
            alpha=0.5,
            linewidth=1,
        )
        plt.title(f"Channel {j} Sin")
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)

    plt.figure(figsize=(24, 12))
    for j in range(phase_channels):
        plt.subplot(2, 4, j + 1)
        plt.plot(
            np.arange(current_frame, current_frame + window_size),
            cos_history[:, j],
            alpha=0.5,
            linewidth=1,
        )
        plt.title(f"Channel {j} Cos")
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    config_path = "humanoid_loco/scripts/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    seed = config["seed"]
    n_joints = config["n_joints"]
    intermediate_channels = config["intermediate_channels"]
    phase_channels = config["phase_channels"]
    total_frames = int(config["fps"] * config["window"]) + 1
    sampled_frames = config["MANN"]["sampled_frames"]
    validation_ratio = config["validation_ratio"]
    datapath = config["datapath"]
    ckptpath = config["ckptpath"]

    dataset = DatasetMANN(
        seed,
        n_joints,
        intermediate_channels,
        phase_channels,
        total_frames,
        sampled_frames,
        validation_ratio,
        datapath,
        ckptpath,
    )
    dataset.set_mode("total")

    validate_dataset(dataset)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    window_size = 1000

    first_frame = 0
    # plot_cmd_vels(dataloader, first_frame, window_size)
    # plot_character(dataloader, first_frame, window_size)
    plot_phases(dataloader, sampled_frames, window_size)

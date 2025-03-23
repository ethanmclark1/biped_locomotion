import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

import biped_locomotion.scripts.utils.helper as helper

from datetime import datetime


def plot_continuous(
    ax: plt.Axes,
    target: torch.Tensor,
    predicted: torch.Tensor,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    ylim: tuple = None,
) -> None:

    target = helper.item(target)
    predicted = helper.item(predicted)

    mse = F.mse_loss(predicted, target)

    ax.plot(target, label="Target", alpha=0.7)
    ax.plot(predicted, "--", label="Predicted", alpha=0.7)
    if title:
        ax.set_title(f"{title}, MSE: {mse:.4f}")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_binary(
    ax: plt.Axes,
    target: torch.Tensor,
    predicted: torch.Tensor,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    time = np.arange(target.shape[0])

    target = helper.item(target)
    predicted = helper.item(predicted)

    ax.step(time, target, where="post", label="Target", alpha=0.7)
    ax.step(time, predicted, where="post", label="Predicted", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    ax.legend()


def save_img(
    fig: plt.Figure,
    axes: np.ndarray,
    n_components: int,
    imgpath: str,
    filename: str = None,
) -> None:
    for ax in axes[n_components:]:
        fig.delaxes(ax)

    plt.tight_layout()

    if not filename:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{now}.png"

    filepath = os.path.join(imgpath, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_results(
    imgpath: str,
    counter: int,
    n_joints: int,
    sampled_frames: int,
    phase_channels: int,
    predicted_robot_projected_gravity: torch.Tensor = None,
    predicted_robot_oriented_xy_vel: torch.Tensor = None,
    predicted_robot_oriented_yaw_vel: torch.Tensor = None,
    predicted_joint_positions: torch.Tensor = None,
    predicted_joint_velocities: torch.Tensor = None,
    predicted_foot_contacts: torch.Tensor = None,
    predicted_phase_updates: torch.Tensor = None,
    target_robot_projected_gravity: torch.Tensor = None,
    target_robot_oriented_xy_vel: torch.Tensor = None,
    target_robot_oriented_yaw_vel: torch.Tensor = None,
    target_joint_positions: torch.Tensor = None,
    target_joint_velocities: torch.Tensor = None,
    target_foot_contacts: torch.Tensor = None,
    target_phase_updates: torch.Tensor = None,
) -> None:

    # Plot 1: Root Projected Gravity
    if (
        target_robot_projected_gravity is not None
        and predicted_robot_projected_gravity is not None
    ):
        dims = target_robot_projected_gravity.shape[1]
        plt.close("all")
        fig, axes = plt.subplots(dims, 1, figsize=(12, 10))
        axes = axes.flatten()

        values = "XYZ"
        for i in range(dims):
            var = values[i]
            plot_continuous(
                axes[i],
                target=target_robot_projected_gravity[:, i : i + 1],
                predicted=predicted_robot_projected_gravity[:, i : i + 1],
                title=f"Robot Projected {var} Gravity",
                xlabel="Frame",
                ylabel="Gravity",
            )

        save_img(
            fig, axes, dims, imgpath, filename=f"{counter}_robot_projected_gravity"
        )

    # Plot 2: World-Oriented velocities
    if (
        target_robot_oriented_xy_vel is not None
        and target_robot_oriented_yaw_vel is not None
        and predicted_robot_oriented_xy_vel is not None
        and predicted_robot_oriented_yaw_vel is not None
    ):
        plt.close("all")
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        axes = axes.flatten()

        variables = "XY"
        for idx in range(target_robot_oriented_xy_vel.shape[1]):
            var = variables[idx]
            plot_continuous(
                axes[idx],
                target_robot_oriented_xy_vel[:, idx],
                predicted_robot_oriented_xy_vel[:, idx],
                f"Robot-Oriented Linear {var} Velocity",
                "Frame",
                "Velocity",
            )

        plot_continuous(
            axes[2],
            target_robot_oriented_yaw_vel,
            predicted_robot_oriented_yaw_vel,
            "Robot-Oriented Yaw Velocity",
            "Frame",
            "Velocity",
        )

        save_img(fig, axes, 3, imgpath, filename=f"{counter}_robot_oriented_velocities")

    # Plot 3 & 4: Joint Positions & Joint Velocities
    if (
        target_joint_positions is not None
        and predicted_joint_positions is not None
        and target_joint_velocities is not None
        and predicted_joint_velocities is not None
    ):
        target_joint_states = [
            target_joint_positions,
            target_joint_velocities,
        ]
        predicted_joint_states = [
            predicted_joint_positions,
            predicted_joint_velocities,
        ]

        for i, (target, predicted) in enumerate(
            zip(target_joint_states, predicted_joint_states)
        ):
            plt.close("all")
            fig, axes = plt.subplots(6, 4, figsize=(20, 15))
            axes = axes.flatten()
            for j in range(n_joints):
                joint_idx = j
                component_name = "Position" if i == 0 else "Velocity"
                plot_continuous(
                    axes[joint_idx],
                    target[:, joint_idx],
                    predicted[:, joint_idx],
                    f"Joint {component_name} {joint_idx}",
                    "Frame",
                    "Angle",
                )

            save_img(
                fig,
                axes,
                n_joints,
                imgpath,
                filename=f"{counter}_joint_{component_name.lower()}",
            )

    # Plot 5: Foot Contacts
    if target_foot_contacts is not None and predicted_foot_contacts is not None:
        plt.close("all")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = axes.flatten()

        variables = ["Right", "Left"]
        for idx in range(target_foot_contacts.shape[1]):
            name = variables[idx]
            plot_binary(
                axes[idx],
                target_foot_contacts[:, idx],
                predicted_foot_contacts[:, idx],
                title=f"{name} Foot Contact",
                xlabel="Frame",
                ylabel="Contact",
            )

        save_img(fig, axes, 2, imgpath, filename=f"{counter}_foot_contacts")

    # Plot 6: Phase Updates
    if target_phase_updates is not None and predicted_phase_updates is not None:
        plt.close("all")
        fig, axes = plt.subplots(2, 2, figsize=(12, 5))
        axes = axes.flatten()

        target_phase_updates = target_phase_updates.reshape(
            -1, sampled_frames // 2 + 1, phase_channels, 4
        )
        predicted_phase_updates = predicted_phase_updates.reshape(
            -1, sampled_frames // 2 + 1, phase_channels, 4
        )

        parameters = ["Sine", "Cosine", "Amplitude", "Frequency"]
        for i, component in enumerate(parameters):
            plt.close("all")
            fig, axes = plt.subplots(2, 4, figsize=(20, 15))
            axes = axes.flatten()
            for channel in range(phase_channels):
                plot_continuous(
                    axes[channel],
                    target_phase_updates[:, sampled_frames // 2, channel, i],
                    predicted_phase_updates[:, sampled_frames // 2, channel, i],
                    f"Channel {channel} {component}",
                    "Frame",
                    "Value",
                )

            save_img(
                fig,
                axes,
                phase_channels,
                imgpath,
                filename=f"{counter}_phase_updates_{component.lower()}",
            )

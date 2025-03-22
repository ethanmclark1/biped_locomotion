import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

import kinematic_predictor.scripts.utils.helper as helper
import kinematic_predictor.scripts.utils.plotter as plotter

from kinematic_predictor.scripts.mann.dataset import *
from kinematic_predictor.scripts.mann.network import MANN

from torch.utils.data import DataLoader


class DeepPhase:
    def __init__(
        self,
        fps: int,
        phase_channels: int,
        total_frames: int,
        sampled_frames: int,
        device: str,
    ) -> None:

        self.dt = 1 / fps
        self.phase_channels = phase_channels
        self.total_frames = total_frames
        self.sampled_frames = sampled_frames
        self.device = device

        self._phase_angle = torch.zeros(
            (self.total_frames, self.phase_channels), device=device, dtype=torch.float32
        )
        self._amplitudes = torch.zeros(
            (self.total_frames, self.phase_channels), device=device, dtype=torch.float32
        )
        self._frequencies = torch.zeros(
            (self.total_frames, self.phase_channels), device=device, dtype=torch.float32
        )

    @property
    def phase_angle(self) -> torch.Tensor:
        return self._phase_angle

    @property
    def amplitudes(self) -> torch.Tensor:
        return self._amplitudes

    @property
    def frequencies(self) -> torch.Tensor:
        return self._frequencies

    # convert polar angle to phase vector (sin, cos)
    def _get_phase_vector(self, phase: torch.Tensor) -> torch.Tensor:
        phase = phase * 2 * np.pi
        return torch.stack([torch.sin(phase), torch.cos(phase)])

    # calculate the angle between two vectors
    def _get_angle(
        self, from_vec: torch.Tensor, to_vec: torch.Tensor, tol: float = 1e-15
    ) -> torch.Tensor:

        epsilon = torch.sqrt((from_vec * from_vec).sum() * (to_vec * to_vec).sum())
        if epsilon < tol:
            return torch.tensor(0.0)

        dot = torch.clamp(from_vec @ to_vec / epsilon, -1.0, 1.0)
        angle = torch.acos(dot) * 180 / np.pi
        return angle

    def _get_signed_angle(
        self, from_vec: torch.Tensor, to_vec: torch.Tensor
    ) -> torch.Tensor:

        sign = torch.sign(from_vec[0] * to_vec[1] - from_vec[1] * to_vec[0])
        angle = self._get_angle(from_vec, to_vec)

        return sign * angle

    # convert phase vector (sin, cos) to a polar angle in degrees
    def _get_phase_value(self, phase_vec: torch.Tensor) -> torch.Tensor:
        if torch.linalg.norm(phase_vec) == 0:
            return torch.zeros_like(phase_vec[0:1])

        up = torch.tensor([0.0, 1.0], device=phase_vec.device)
        phase_norm = phase_vec / torch.linalg.norm(phase_vec)
        angle = -self._get_signed_angle(up, phase_norm)

        if angle < 0:
            angle = 360 + angle

        angle = angle / 360 % 1

        return angle

    # calculate the phase angle update between two phase vectors and normalize it
    def _get_phase_update(
        self, from_angle: torch.Tensor, to_angle: torch.Tensor
    ) -> torch.Tensor:

        # intentionally converting to phase angle before this since phase_vector and phase_value are not inverses of each other
        from_vec = self._get_phase_vector(from_angle)
        to_vec = self._get_phase_vector(to_angle)
        # same operation as phase value except we're passing in two vectors instead of using up vector
        signed_angle = -self._get_signed_angle(from_vec, to_vec)
        # normalization to ensure the angle is between 0 and 1
        if signed_angle != 0:
            signed_angle = signed_angle / 360 % 1

        return signed_angle

    def increment_phase_series(self) -> None:
        current_total_frame = self.total_frames // 2

        self._phase_angle[0:current_total_frame] = self._phase_angle[
            1 : current_total_frame + 1
        ].clone()
        self._amplitudes[0:current_total_frame] = self._amplitudes[
            1 : current_total_frame + 1
        ].clone()
        self._frequencies[0:current_total_frame] = self._frequencies[
            1 : current_total_frame + 1
        ].clone()

    def get_phase_space(self) -> torch.Tensor:
        adjusted_phase_space = torch.zeros(
            (self.sampled_frames, self.phase_channels, 2)
        )
        for i in range(self.sampled_frames):
            total_index = i * 10
            for j in range(self.phase_channels):
                phase_vec = self._get_phase_vector(self._phase_angle[total_index, j])
                # multiply by amplitude to zero out frames that have not been experienced hence why amplitude is 0
                adjusted_phase_space[i, j] = (
                    self._amplitudes[total_index, j] * phase_vec
                )

        phase_space = adjusted_phase_space.reshape(1, -1)

        return phase_space.to(self.device)

    def combine_phases(self, phase_update: torch.Tensor) -> torch.Tensor:
        pivot = 0
        phase_update = phase_update.squeeze(0)
        current_sampled_frame = self.sampled_frames // 2

        updated_phase_series = torch.zeros(
            (self.sampled_frames // 2 + 1, self.phase_channels, 3)
        )

        for sampled_frame_idx in range(current_sampled_frame, self.sampled_frames):
            # phase angle is entire 121 frames
            index = sampled_frame_idx * 10
            k = sampled_frame_idx - current_sampled_frame
            for channel_idx in range(self.phase_channels):
                cur_phase_val = self._phase_angle[index, channel_idx]
                cur_phase_vec = self._get_phase_vector(cur_phase_val)

                predicted_phase_vec = torch.tensor(
                    [phase_update[pivot + 0], phase_update[pivot + 1]]
                )
                predicted_amp = torch.abs(phase_update[pivot + 2])
                predicted_freq = torch.abs(phase_update[pivot + 3])
                pivot += 4

                # authors multiply by 360 instead of 2*np.pi because Quaternion.AngleAxis expects angle in degrees
                theta = -predicted_freq * 2 * np.pi * self.dt
                cos_theta = torch.cos(theta)
                sin_theta = torch.sin(theta)

                x = cur_phase_vec[0] * cos_theta - cur_phase_vec[1] * sin_theta
                y = cur_phase_vec[0] * sin_theta + cur_phase_vec[1] * cos_theta
                updated_phase_vec = torch.tensor([x, y])

                # normalize the phase vectors
                updated_phase_vec = updated_phase_vec / torch.linalg.norm(
                    updated_phase_vec
                )
                predicted_phase_vec = predicted_phase_vec / torch.linalg.norm(
                    predicted_phase_vec
                )

                t = 0.5
                lerped = (1.0 - t) * updated_phase_vec + t * predicted_phase_vec
                lerped = lerped / torch.linalg.norm(lerped)

                adjusted_phase = self._get_phase_value(lerped.float())

                # adjusted phase is degrees normalized by 360 which is then used to create phase vector
                updated_phase_series[k, channel_idx, 0] = adjusted_phase
                updated_phase_series[k, channel_idx, 1] = predicted_amp
                updated_phase_series[k, channel_idx, 2] = predicted_freq

        return updated_phase_series

    def interpolate_phase_series(
        self, updated_phase_series: torch.Tensor
    ) -> torch.Tensor:
        current_total_frame = self.total_frames // 2

        weights = torch.linspace(0, 0.9, steps=10)

        interpolated_phase_series = torch.zeros(
            (self.total_frames // 2 + 1, self.phase_channels, 3)
        )

        for frame_idx in range(current_total_frame, self.total_frames):
            increment = frame_idx % 10
            weight = weights[increment].item()

            if weight == 0:
                # ensure no interpolation performed at sampled frames [60,70,80,90,100,110,120]
                prev_index = next_index = int(frame_idx / 10) - self.sampled_frames // 2

            interpolated_idx = frame_idx - self.total_frames // 2
            for channel_idx in range(self.phase_channels):
                prev_phase_angle = updated_phase_series[prev_index, channel_idx, 0]
                prev_amp = updated_phase_series[prev_index, channel_idx, 1]
                prev_freq = updated_phase_series[prev_index, channel_idx, 2]

                next_phase_angle = updated_phase_series[next_index, channel_idx, 0]
                next_amp = updated_phase_series[next_index, channel_idx, 1]
                next_freq = updated_phase_series[next_index, channel_idx, 2]

                scaled_prev_phase_vec = prev_amp * self._get_phase_vector(
                    prev_phase_angle
                )
                scaled_next_phase_vec = next_amp * self._get_phase_vector(
                    next_phase_angle
                )

                update = self._get_phase_update(
                    self._get_phase_value(scaled_prev_phase_vec),
                    self._get_phase_value(scaled_next_phase_vec),
                )

                theta = -update * 2 * np.pi * weight
                cos_theta = torch.cos(theta)
                sin_theta = torch.sin(theta)

                x = (
                    scaled_prev_phase_vec[0] * cos_theta
                    - scaled_prev_phase_vec[1] * sin_theta
                )
                y = (
                    scaled_prev_phase_vec[0] * sin_theta
                    + scaled_prev_phase_vec[1] * cos_theta
                )

                lerped_phase_vec = torch.tensor([x, y])
                lerped_phase_angle = self._get_phase_value(lerped_phase_vec)
                lerped_amp = torch.lerp(prev_amp, next_amp, weight)
                lerped_freq = torch.lerp(prev_freq, next_freq, weight)

                interpolated_phase_series[
                    interpolated_idx, channel_idx, 0
                ] = lerped_phase_angle
                interpolated_phase_series[interpolated_idx, channel_idx, 1] = lerped_amp
                interpolated_phase_series[
                    interpolated_idx, channel_idx, 2
                ] = lerped_freq

            if weight == 0:
                next_index += 1

        self._phase_angle[current_total_frame:] = interpolated_phase_series[..., 0]
        self._amplitudes[current_total_frame:] = interpolated_phase_series[..., 1]
        self._frequencies[current_total_frame:] = interpolated_phase_series[..., 2]


if __name__ == "__main__":
    """
    --------------------------------------------------------------------------
    Initializations
    """
    config_path = "kinematic_predictor/scripts/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    fps = config["fps"]
    window = config["window"]
    total_frames = int(fps * window) + 1
    n_joints = config["n_joints"]
    intermediate_channels = config["intermediate_channels"]
    phase_channels = config["phase_channels"]

    full_joint_state = config["PAE"]["full_joint_state"]

    n_epochs = config["MANN"]["n_epochs"]
    batch_size = config["MANN"]["batch_size"]
    sampled_frames = config["MANN"]["sampled_frames"]

    window_size = 300

    seed = config["seed"]
    helper.set_seed(seed)

    device = helper.get_device()

    datapath = config["datapath"]
    ckptpath = config["ckptpath"]
    imgpath = config["imgpath"]
    imgpath = os.path.join(imgpath, "inference")
    version_no = config["version_no"]

    ckpt = os.path.join(
        ckptpath,
        f"mann_{version_no}_{phase_channels}phases_{intermediate_channels}intermediate_{total_frames}frames_{full_joint_state}.pt",
    )

    imgpath = os.path.join(
        imgpath,
        f"{version_no}_{phase_channels}phases_{intermediate_channels}intermediate_{total_frames}frames_{full_joint_state}",
    )
    os.makedirs(imgpath, exist_ok=True)

    """
    --------------------------------------------------------------------------
    Create Dataset & DataLoader
    """
    dataset = DatasetMANN(
        seed,
        n_joints,
        intermediate_channels,
        phase_channels,
        total_frames,
        sampled_frames,
        0.0,
        datapath,
        ckptpath,
        version_no,
    )
    dataset.set_mode("total")

    n_samples = len(dataset)

    x_norm = dataset.x_norm
    y_norm = dataset.y_norm

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed),
    )

    """
    --------------------------------------------------------------------------
    Initialize Mode-Adaptive Neural Network
    """
    mann = helper.to_device(
        MANN(
            x_norm,
            y_norm,
            batch_size,
            n_samples,
            config,
        )
    )

    try:
        ckpt = torch.load(ckpt, map_location=device)
        mann.load_state_dict(ckpt, strict=False)
        mann.eval()
    except FileNotFoundError:
        raise FileNotFoundError("MANN model checkpoint not found")

    """
    --------------------------------------------------------------------------
    Initialize DeepPhase
    """
    deep_phase = DeepPhase(fps, phase_channels, total_frames, sampled_frames, device)

    """
    --------------------------------------------------------------------------
    Plot phases for analysis
    """

    def plot_phases(
        ground_truth_phase_space: torch.Tensor, adjusted_phase_space: torch.Tensor
    ) -> None:

        ground_truth_phase_space = (
            ground_truth_phase_space.reshape(-1, sampled_frames, phase_channels, 2)
            .cpu()
            .numpy()
            .squeeze(0)
        )
        adjusted_phase_space = (
            adjusted_phase_space.reshape(-1, sampled_frames, phase_channels, 2)
            .cpu()
            .numpy()
            .squeeze(0)
        )

        plt.figure(figsize=(24, 12))
        for j in range(phase_channels):
            ax = plt.subplot(2, 4, j + 1)
            plt.plot(
                np.arange(-sampled_frames // 2 + 1, sampled_frames // 2 + 1) * 10,
                adjusted_phase_space[:, j, 0],
                alpha=0.5,
                linewidth=1,
                label="Predicted",
            )
            plt.plot(
                np.arange(-sampled_frames // 2 + 1, sampled_frames // 2 + 1) * 10,
                ground_truth_phase_space[:, j, 0],
                alpha=0.5,
                linewidth=1,
                label="Ground Truth",
            )
            plt.title(f"Channel {j} Sin")
            plt.grid(True, alpha=0.3)
            ax.legend()
        plt.xlabel("Gather Window")
        plt.tight_layout()
        plt.show(block=True)

        plt.figure(figsize=(24, 12))
        for j in range(phase_channels):
            ax = plt.subplot(2, 4, j + 1)
            plt.plot(
                np.arange(-sampled_frames // 2 + 1, sampled_frames // 2 + 1) * 10,
                adjusted_phase_space[:, j, 1],
                alpha=0.5,
                linewidth=1,
                label="Predicted",
            )
            plt.plot(
                np.arange(-sampled_frames // 2 + 1, sampled_frames // 2 + 1) * 10,
                ground_truth_phase_space[:, j, 1],
                alpha=0.5,
                linewidth=1,
                label="Ground Truth",
            )
            plt.title(f"Channel {j} Cos")
            plt.grid(True, alpha=0.3)
            ax.legend()
        plt.xlabel("Gather Window")
        plt.tight_layout()
        plt.show(block=True)

    """
    --------------------------------------------------------------------------
    Inference Loop for Mode-Adaptive Network
    """
    predicted_robot_projected_gravity = []
    predicted_robot_oriented_xy_vel = []
    predicted_robot_oriented_yaw_vel = []
    predicted_joint_positions = []
    predicted_joint_velocities = []
    predicted_foot_contacts = []
    predicted_phase_update = []

    target_robot_projected_gravity = []
    target_robot_oriented_xy_vel = []
    target_robot_oriented_yaw_vel = []
    target_joint_positions = []
    target_joint_velocities = []
    target_foot_contacts = []
    target_phase_update = []

    input = dataloader.dataset[0][0].unsqueeze(0)
    # TODO: Figure out how to get the initial cmd_vels
    cmd_vels_linear_xy = input[:, INPUT_STRUCTURE["cmd_vels_linear_xy"]]
    cmd_vels_yaw = input[:, INPUT_STRUCTURE["cmd_vels_yaw"]]
    robot_projected_gravity = input[:, INPUT_STRUCTURE["robot_projected_gravity"]]
    robot_oriented_xy_vel = input[:, INPUT_STRUCTURE["robot_oriented_xy_vel"]]
    robot_oriented_yaw_vel = input[:, INPUT_STRUCTURE["robot_oriented_yaw_vel"]]
    joint_positions = input[:, INPUT_STRUCTURE["joint_positions"]]
    joint_velocities = input[:, INPUT_STRUCTURE["joint_velocities"]]
    foot_contacts = input[:, INPUT_STRUCTURE["foot_contacts"]]

    for i, (_, target) in enumerate(dataloader):
        if i == window_size:
            break

        # TODO: Optimize efficiency of this as it uses nested for loops
        phase_space = deep_phase.get_phase_space()

        input = torch.cat(
            (
                cmd_vels_linear_xy,
                cmd_vels_yaw,
                robot_projected_gravity,
                robot_oriented_xy_vel,
                robot_oriented_yaw_vel,
                joint_positions,
                joint_velocities,
                foot_contacts,
                phase_space,
            ),
            axis=1,
        )

        with torch.no_grad():
            y_pred = mann(input)

        robot_projected_gravity = y_pred[:, TARGET_STRUCTURE["robot_projected_gravity"]]
        robot_oriented_xy_vel = y_pred[:, TARGET_STRUCTURE["robot_oriented_xy_vel"]]
        robot_oriented_yaw_vel = y_pred[:, TARGET_STRUCTURE["robot_oriented_yaw_vel"]]
        joint_positions = y_pred[:, TARGET_STRUCTURE["joint_positions"]]
        joint_velocities = y_pred[:, TARGET_STRUCTURE["joint_velocities"]]
        foot_contacts = y_pred[:, TARGET_STRUCTURE["foot_contacts"]]
        phase_update = y_pred[:, TARGET_STRUCTURE["phase_update"]]

        ground_truth_next_input = dataloader.dataset[i + 1][0].unsqueeze(0)
        cmd_vels_linear_xy = ground_truth_next_input[
            :, INPUT_STRUCTURE["cmd_vels_linear_xy"]
        ]
        cmd_vels_yaw = ground_truth_next_input[:, INPUT_STRUCTURE["cmd_vels_yaw"]]
        ground_truth_phase_space = ground_truth_next_input[
            :, INPUT_STRUCTURE["phase_space"]
        ]

        # TODO: Optimize efficiency of these as they all use nested for loops
        deep_phase.increment_phase_series()
        updated_phase_series = deep_phase.combine_phases(phase_update)
        deep_phase.interpolate_phase_series(updated_phase_series)

        predicted_robot_projected_gravity.append(robot_projected_gravity)
        predicted_robot_oriented_xy_vel.append(robot_oriented_xy_vel)
        predicted_robot_oriented_yaw_vel.append(robot_oriented_yaw_vel)
        predicted_joint_positions.append(joint_positions)
        predicted_joint_velocities.append(joint_velocities)
        predicted_foot_contacts.append(foot_contacts)
        predicted_phase_update.append(phase_update)

        target_robot_projected_gravity.append(
            target[:, TARGET_STRUCTURE["robot_projected_gravity"]]
        )
        target_robot_oriented_xy_vel.append(
            target[:, TARGET_STRUCTURE["robot_oriented_xy_vel"]]
        )
        target_robot_oriented_yaw_vel.append(
            target[:, TARGET_STRUCTURE["robot_oriented_yaw_vel"]]
        )
        target_joint_positions.append(target[:, TARGET_STRUCTURE["joint_positions"]])
        target_joint_velocities.append(target[:, TARGET_STRUCTURE["joint_velocities"]])
        target_foot_contacts.append(target[:, TARGET_STRUCTURE["foot_contacts"]])
        target_phase_update.append(target[:, TARGET_STRUCTURE["phase_update"]])

    predicted_robot_projected_gravity = torch.cat(
        predicted_robot_projected_gravity, dim=0
    )
    predicted_robot_oriented_xy_vel = torch.cat(predicted_robot_oriented_xy_vel, dim=0)
    predicted_robot_oriented_yaw_vel = torch.cat(
        predicted_robot_oriented_yaw_vel, dim=0
    )
    predicted_joint_positions = torch.cat(predicted_joint_positions, dim=0)
    predicted_joint_velocities = torch.cat(predicted_joint_velocities, dim=0)
    predicted_foot_contacts = torch.cat(predicted_foot_contacts, dim=0)
    predicted_phase_update = torch.cat(predicted_phase_update, dim=0)

    target_robot_projected_gravity = torch.cat(target_robot_projected_gravity, dim=0)
    target_robot_oriented_xy_vel = torch.cat(target_robot_oriented_xy_vel, dim=0)
    target_robot_oriented_yaw_vel = torch.cat(target_robot_oriented_yaw_vel, dim=0)
    target_joint_positions = torch.cat(target_joint_positions, dim=0)
    target_joint_velocities = torch.cat(target_joint_velocities, dim=0)
    target_foot_contacts = torch.cat(target_foot_contacts, dim=0)
    target_phase_update = torch.cat(target_phase_update, dim=0)

    plotter.plot_results(
        imgpath=imgpath,
        counter=i,
        n_joints=n_joints,
        sampled_frames=sampled_frames,
        phase_channels=phase_channels,
        predicted_robot_projected_gravity=predicted_robot_projected_gravity,
        predicted_robot_oriented_xy_vel=predicted_robot_oriented_xy_vel,
        predicted_robot_oriented_yaw_vel=predicted_robot_oriented_yaw_vel,
        predicted_joint_positions=predicted_joint_positions,
        predicted_joint_velocities=predicted_joint_velocities,
        predicted_foot_contacts=predicted_foot_contacts,
        predicted_phase_updates=predicted_phase_update,
        target_robot_projected_gravity=target_robot_projected_gravity,
        target_robot_oriented_xy_vel=target_robot_oriented_xy_vel,
        target_robot_oriented_yaw_vel=target_robot_oriented_yaw_vel,
        target_joint_positions=target_joint_positions,
        target_joint_velocities=target_joint_velocities,
        target_foot_contacts=target_foot_contacts,
        target_phase_updates=target_phase_update,
    )

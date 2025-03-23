import torch
import numpy as np

import biped_locomotion.scripts.utils.helper as helper

from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R


GRAVITY = np.array([0, 0, -9.81])

INPUT_STRUCTURE = {
    "cmd_vels_linear_xy": slice(0, 26),
    "cmd_vels_yaw": slice(26, 39),
    "robot_projected_gravity": slice(39, 42),
    "robot_oriented_xy_vel": slice(42, 44),
    "robot_oriented_yaw_vel": slice(44, 45),
    "joint_positions": slice(45, 68),
    "joint_velocities": slice(68, 91),
    "foot_contacts": slice(91, 93),
    "phase_space": slice(93, 301),
}

TARGET_STRUCTURE = {
    "robot_projected_gravity": slice(0, 3),
    "robot_oriented_xy_vel": slice(3, 5),
    "robot_oriented_yaw_vel": slice(5, 6),
    "joint_positions": slice(6, 29),
    "joint_velocities": slice(29, 52),
    "foot_contacts": slice(52, 54),
    "phase_update": slice(54, 278),
}


class DatasetMANN(Dataset):
    def __init__(
        self,
        seed: int,
        n_joints: int,
        intermediate_channels: int,
        phase_channels: int,
        total_frames: int,
        sampled_frames: int,
        validation_ratio: float,
        datapath: str,
        ckptpath: str,
        full_joint_state: bool = False,
        mirroring: bool = False,
    ) -> None:
        super(DatasetMANN, self).__init__()

        torch.manual_seed(seed)

        self.mode = None
        self.input_data = None
        self.target_data = None
        self.device = helper.get_device()

        data_paths = helper.get_data_paths(datapath)
        self._process_data(
            data_paths,
            n_joints,
            intermediate_channels,
            phase_channels,
            total_frames,
            sampled_frames,
            ckptpath,
            full_joint_state,
        )

        num_samples = len(self.input_data) - 1

        self.total_indices = torch.arange(num_samples, device=self.device)

        self.eval_size = int(num_samples * validation_ratio)
        self.train_indices = self.total_indices[: -self.eval_size]
        self.eval_indices = self.total_indices[-self.eval_size + total_frames - 1 :]

    def __len__(self) -> int:
        if self.mode == "train":
            length = len(self.train_indices)
        elif self.mode == "eval":
            length = len(self.eval_indices)
        elif self.mode == "total":
            length = len(self.total_indices)
        else:
            raise ValueError("Mode must be 'train', 'eval', or 'total'")

        return length

    def __getitem__(self, mode_idx: int) -> tuple:
        # get the actual index from the current mode indices
        actual_idx = self.mode_indices[mode_idx]

        input_item = self.input_data[actual_idx]
        target_item = self.target_data[actual_idx]

        return input_item, target_item

    @property
    def input_shape(self) -> tuple:
        return self.input_data.shape[1]

    @property
    def target_shape(self) -> tuple:
        return self.target_data.shape[1]

    @property
    def x_norm(self) -> torch.Tensor:
        mean = self.input_data[:, :].mean(axis=0)
        std = self.input_data[:, :].std(axis=0)
        return torch.vstack((mean, std)).float()

    @property
    def y_norm(self) -> tuple:
        mean = self.target_data[:, :].mean(axis=0)
        std = self.target_data[:, :].std(axis=0)
        return torch.vstack((mean, std)).float()

    def _process_character_data(
        self,
        root_states: torch.Tensor,
        joint_states: torch.Tensor,
        foot_contacts: torch.Tensor,
        n_joints: int,
        total_frames: int,
    ) -> tuple:

        root_position = root_states[:, :3]
        root_quaternion = root_states[:, [3, 4, 5, 6]]  # (w,x,y,z)
        forward_pos = np.zeros((root_position.shape[0] - 2 * total_frames, 13))
        sideways_pos = np.zeros((root_position.shape[0] - 2 * total_frames, 13))
        yaw_pos = np.zeros((root_position.shape[0] - 2 * total_frames, 13))

        valid_frames_start = total_frames
        valid_frames_end = root_position.shape[0] - total_frames

        for i in range(valid_frames_start, valid_frames_end):
            # get the index of the first frame in the window
            k = i - valid_frames_start
            # get gather window of 13 frames, centered at i
            position_window = root_position[k : k + total_frames : 10]
            quaternion_window = root_quaternion[k : k + total_frames : 10]

            # get origin position and orientation
            origin = position_window[7]
            origin_z_angle = R.from_quat(quaternion_window[7]).as_euler("zyx")[0]

            # get xy positions relative to origin
            relative_positions = position_window - origin
            forward = relative_positions[:, 0] * np.cos(
                -origin_z_angle
            ) - relative_positions[:, 1] * np.sin(-origin_z_angle)
            sideways = relative_positions[:, 0] * np.sin(
                -origin_z_angle
            ) + relative_positions[:, 1] * np.cos(-origin_z_angle)

            z_angles = R.from_quat(quaternion_window).as_euler("zyx")[:, 0]

            forward_pos[k] = forward
            sideways_pos[k] = sideways
            yaw_pos[k] = z_angles - origin_z_angle

        forward_vel = np.gradient(forward_pos, axis=1)
        sideways_vel = np.gradient(sideways_pos, axis=1)
        # unwrap yaw to get continuous values
        yaw_vel = np.gradient(np.unwrap(yaw_pos), axis=1)
        cmd_vels = torch.from_numpy(np.hstack((forward_vel, sideways_vel, yaw_vel)))

        root_rotations = R.from_quat(root_quaternion)
        inverse_root_rotation = root_rotations.inv()
        robot_projected_gravity = torch.from_numpy(inverse_root_rotation.apply(GRAVITY))

        z_angle = root_rotations.as_euler("zyx")[:, 0]
        x_vel = root_states[:, 7]
        y_vel = root_states[:, 8]
        forward_vel = x_vel * np.cos(-z_angle) + y_vel * np.sin(-z_angle)
        sideways_vel = x_vel * np.sin(-z_angle) + y_vel * np.cos(-z_angle)
        robot_oriented_xy_linear_vel = torch.stack((forward_vel, sideways_vel), axis=1)

        robot_oriented_yaw_velocity = root_states[:, 12].reshape(-1, 1)

        joint_positions = joint_states[:, :n_joints]
        joint_velocities = joint_states[:, n_joints:]

        # skip first and last 121 frames to match phase space
        character_data = torch.hstack(
            (
                robot_projected_gravity,
                robot_oriented_xy_linear_vel,
                robot_oriented_yaw_velocity,
                joint_positions,
                joint_velocities,
                foot_contacts,
            )
        )[valid_frames_start:valid_frames_end]

        return (cmd_vels, character_data)

    """
    Parameters file has root_state.shape[0] - total_frames
    """

    def _process_phase_space_data(
        self,
        parameters: torch.Tensor,
        phase_channels: int,
        total_frames: int,
        sampled_frames: int,
    ) -> torch.Tensor:

        pad_size = total_frames // 2

        valid_frames_start = pad_size
        valid_frames_end = parameters.shape[0] - pad_size - 1

        n_samples = parameters.shape[0] - total_frames

        # shift is normalized between [0,1]
        shift = parameters[:, :phase_channels]
        amp = parameters[:, 2 * phase_channels : 3 * phase_channels]

        phase_space = torch.zeros((n_samples, phase_channels, sampled_frames, 2))

        for i in range(valid_frames_start, valid_frames_end):
            k = i - pad_size
            # get gather window of 121 frames, centered at i
            window_shift = shift[k : k + total_frames : 10]
            window_amp = amp[k : k + total_frames : 10]

            # transpose to match shape of phase_space (n_samples, n_channels, sampled_frames, 2)
            phase_space[k, ..., 0] = (window_amp * np.sin(2 * np.pi * window_shift)).T
            phase_space[k, ..., 1] = (window_amp * np.cos(2 * np.pi * window_shift)).T

        # reshape to (n_samples, sampled_frames, n_channels, 2)
        phase_space = phase_space.swapaxes(1, 2)
        return phase_space.reshape(n_samples, -1)

    def _process_phase_update_data(
        self,
        parameters: torch.Tensor,
        phase_channels: int,
        total_frames: int,
        sampled_frames: int,
    ) -> torch.Tensor:

        pad_size = total_frames // 2

        valid_frames_start = pad_size
        valid_frames_end = parameters.shape[0] - pad_size - 1

        n_samples = parameters.shape[0] - total_frames

        forward_sample = sampled_frames // 2 + 1

        # shift is normalized between [0,1]
        shift = parameters[:, :phase_channels]
        freq = parameters[:, phase_channels : 2 * phase_channels]
        amp = parameters[:, 2 * phase_channels : 3 * phase_channels]

        phase_update = torch.zeros((n_samples, phase_channels, forward_sample, 4))

        for i in range(valid_frames_start, valid_frames_end):
            k = i - pad_size
            # get gather window of 7 frames, beginning at i
            window_shift = shift[i : i + pad_size + 1 : 10]
            window_amp = amp[i : i + pad_size + 1 : 10]
            window_freqs = freq[i : i + pad_size + 1 : 10]

            # transpose to match shape of phase_space (n_samples, sampled_frames//2, n_channels, 4)
            phase_update[k, ..., 0] = (window_amp * np.sin(2 * np.pi * window_shift)).T
            phase_update[k, ..., 1] = (window_amp * np.cos(2 * np.pi * window_shift)).T
            phase_update[k, ..., 2] = window_amp.T
            phase_update[k, ..., 3] = window_freqs.T

        # reshape to (n_samples, sampled_frames // 2, n_channels, 2)
        phase_update = phase_update.swapaxes(1, 2)
        return phase_update.reshape(n_samples, -1)

    def _process_data(
        self,
        data_paths: str,
        n_joints: int,
        intermediate_channels: int,
        phase_channels: int,
        total_frames: int,
        sampled_frames: int,
        ckptpath: str,
        full_joint_state: bool,
        mirroring: bool,
    ) -> None:

        for data_path in data_paths:
            root_states = torch.from_numpy(
                np.load(f"{data_path}_walking_root_states.npy")
            ).to(torch.float32)
            joint_states = torch.from_numpy(
                np.load(f"{data_path}_walking_joint_states.npy") * np.pi / 180
            ).to(torch.float32)
            foot_contacts = torch.from_numpy(
                np.load(f"{data_path}_walking_foot_contacts.npy")
            ).to(torch.float32)
            parameters = torch.from_numpy(
                np.loadtxt(
                    f"{ckptpath}/parameters_{phase_channels}phases_{intermediate_channels}intermediate_{total_frames}frames_{full_joint_state}.txt",
                    dtype=np.float32,
                )
            )

            cmd_vels, character_data = self._process_character_data(
                root_states,
                joint_states,
                foot_contacts,
                n_joints,
                total_frames,
            )
            phase_space = self._process_phase_space_data(
                parameters,
                phase_channels,
                total_frames,
                sampled_frames,
            )
            phase_update = self._process_phase_update_data(
                parameters,
                phase_channels,
                total_frames,
                sampled_frames,
            )


            self.input_data = (
                torch.hstack(
                    (
                        cmd_vels,
                        character_data,
                        phase_space,
                    )
                )[:-1]
                .float()
                .to(self.device)
            )
            self.target_data = (
                torch.hstack((character_data, phase_update))[1:]
                .float()
                .to(self.device)
            )

    def set_mode(self, mode: str) -> None:
        if mode not in ["train", "eval", "total"]:
            raise ValueError("Mode must be 'train', 'eval', or 'total'")

        self.mode = mode
        self.mode_indices = getattr(self, f"{mode}_indices")

import torch
import numpy as np

import humanoid_loco.scripts.utils.helper as helper

from torch.utils.data import Dataset


class DatasetPAE(Dataset):
    def __init__(
        self,
        seed: int,
        n_joints: int,
        frames: int,
        validation_ratio: float,
        datapath: str,
        version_no: str,
        full_joint_state: bool = False,
        mirroring: bool = False,
    ) -> None:

        torch.manual_seed(seed)

        self.mode = None
        self.data = None
        self.device = helper.get_device()
        self.frames = frames

        data_paths = helper.get_data_paths(datapath, version_no)
        self._process_data(data_paths, n_joints, full_joint_state, mirroring)

        gather_window = torch.arange(frames) - (frames // 2)
        self.gather_window = gather_window.reshape(1, -1).to(self.device)

        num_samples = len(self.data)

        self.total_indices = torch.arange(num_samples, device=self.device)

        self.eval_size = int(num_samples * validation_ratio)
        self.train_indices = self.total_indices[self.eval_size :]
        self.eval_indices = self.total_indices[: self.eval_size - frames]

    # subtract frames to only get indices with valid gather windows in the same mode
    def __len__(self) -> int:
        if self.mode == "train":
            length = len(self.train_indices) - self.frames
        elif self.mode == "eval":
            length = len(self.eval_indices) - self.frames
        elif self.mode == "total":
            length = len(self.total_indices) - self.frames
        else:
            raise ValueError("Mode must be 'train', 'eval', or 'total'")

        return length

    def __getitem__(self, mode_idx: int) -> torch.Tensor:
        # actual index in total indices
        actual_idx = self.mode_indices[mode_idx] + self.frames // 2

        # window-based gathering that centers around the actual index
        gather_indices = self.gather_window + actual_idx

        # mask data that doesn't belong to same sequence
        gather_sequence = self.data[gather_indices][..., 0]
        current_sequence = self.data[actual_idx][0]
        sequence_mask = gather_sequence == current_sequence

        # fix for error of "argmax_cuda not implemented for 'Bool'"
        sequence_mask = 1 * sequence_mask
        # find the last sequence-matching index
        last_matching_index = torch.argmax(sequence_mask.flip(dims=[1]), dim=1).to(
            self.device
        )
        last_matching_index = self.gather_window.shape[1] - 1 - last_matching_index

        # duplicate the last matching index to the right
        valid_mask = (
            torch.arange(self.gather_window.shape[1], device=self.gather_window.device)
            <= last_matching_index
        )
        clipped_gather_indices = torch.where(
            valid_mask, gather_indices, gather_indices[:, last_matching_index]
        )

        # drop sequence number at first column
        self.data = self.data
        gathered_data = self.data[clipped_gather_indices].squeeze(0)[:, 1:]

        mean_data = gathered_data.mean(dim=0)
        gathered_data = gathered_data - mean_data

        # swap axes to (n_joints, gather_window)
        gathered_data = gathered_data.swapaxes(0, 1)

        return gathered_data

    # remove sequence number from shape
    @property
    def shape(self) -> tuple:
        return self.data.shape[1] - 1

    def _process_data(
        self,
        data_paths: list,
        n_joints: int,
        full_joint_state: bool,
        mirroring: bool,
    ) -> None:

        for sequence_num, data_path in enumerate(data_paths):
            # convert to radians
            joint_states = torch.from_numpy(
                np.load(f"{data_path}_walking_joint_states.npy") * np.pi / 180
            )

            joint_data = (
                joint_states if full_joint_state else joint_states[:, n_joints:]
            )
            sequence_data = torch.full((joint_data.shape[0], 1), sequence_num)

            data = torch.cat((sequence_data, joint_data), dim=1).float().to(self.device)

            self.data = data if self.data is None else torch.cat((self.data, data))

    def set_mode(self, mode: str) -> None:
        if mode not in ["train", "eval", "total"]:
            raise ValueError("Mode must be 'train', 'eval', or 'total'")

        self.mode = mode
        self.mode_indices = getattr(self, f"{mode}_indices")
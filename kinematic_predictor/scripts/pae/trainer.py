import os
import yaml
import torch
import wandb
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

import kinematic_predictor.scripts.utils.helper as helper
import kinematic_predictor.scripts.utils.plotter as plotter

from torch.utils.data import DataLoader

from kinematic_predictor.scripts.pae.network import PAE
from kinematic_predictor.scripts.pae.dataset import DatasetPAE


class TrainerPAE:
    def __init__(self) -> None:
        self.config = None
        config_path = "kinematic_predictor/scripts/config.yaml"

        self._init_params(config_path)

        self.dataset = DatasetPAE(
            self.seed,
            self.n_joints,
            self.frames,
            self.validation_ratio,
            self.datapath,
            self.version_no,
            self.full_joint_state,
        )

        self.dataset.set_mode("train")
        self.n_samples = len(self.dataset)

        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=0,
        )

        self.input_channels = self.data_loader.dataset.shape

        self.pae = helper.to_device(
            PAE(
                self.input_channels,
                self.intermediate_channels,
                self.phase_channels,
                self.frames,
                self.window,
                self.batch_size,
                self.config,
                self.n_samples,
            )
        )

    def _init_params(self, config_path: str) -> None:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        self.n_joints = config["n_joints"]
        fps = config["fps"]
        self.window = config["window"]
        self.intermediate_channels = config["intermediate_channels"]
        self.phase_channels = config["phase_channels"]

        self.datapath = config["datapath"]
        self.ckptpath = config["ckptpath"]
        imgpath = config["imgpath"]
        imgpath = os.path.join(imgpath, "pae")
        self.version_no = config["version_no"]

        self.frames = int(self.window * fps) + 1
        self.ckpt_period = config["ckpt_period"]
        self.validation_ratio = config["validation_ratio"]

        self.patience = config["PAE"]["patience"]
        self.min_delta = config["PAE"]["min_delta"]
        self.n_epochs = config["PAE"]["n_epochs"]
        self.batch_size = config["PAE"]["batch_size"]
        self.full_joint_state = config["PAE"]["full_joint_state"]

        self.seed = config["seed"]
        helper.set_seed(self.seed)

        self.imgpath = os.path.join(
            imgpath,
            f"{self.version_no}_{self.phase_channels}phases_{self.intermediate_channels}intermediate_{self.frames}frames_{self.full_joint_state}",
        )

        os.makedirs(self.ckptpath, exist_ok=True)
        os.makedirs(self.imgpath, exist_ok=True)

        self.config = config

    def load(self) -> tuple:
        parameters = None

        model_path = f"{self.ckptpath}/pae_{self.version_no}_{self.phase_channels}phases_{self.intermediate_channels}intermediate_{self.frames}frames_{self.full_joint_state}.pt"
        params_path = f"{self.ckptpath}/{self.version_no}_parameters_{self.phase_channels}phases_{self.intermediate_channels}intermediate_{self.frames}frames_{self.full_joint_state}.txt"
        if os.path.exists(model_path) and os.path.exists(params_path):
            checkpoint = torch.load(model_path)
            self.pae.load_state_dict(checkpoint)
            self.pae.eval()
            parameters = np.loadtxt(params_path, dtype=np.float32)
            print("Model and parameters loaded successfully.")
        else:
            raise FileNotFoundError("Model files not found.")

        return parameters

    def init_wandb(self) -> None:
        wandb.init(
            project="Kinematic Predictor",
            entity="ethanmclark1",
            name="Periodic Autoencoder",
        )

        for key, value in self.config.items():
            if isinstance(value, dict):
                if key == "MANN":
                    continue

                for k, v in value.items():
                    wandb.config[k] = v
            else:
                wandb.config[key] = value

        wandb.config["frames"] = self.frames

    def _validate(self) -> float:
        self.pae.eval()
        self.dataset.set_mode("eval")

        total_loss = []
        for batch in self.data_loader:
            with torch.no_grad():
                reconstructed, _ = self.pae(batch)

            loss = self.pae.loss_fn(reconstructed, batch)
            total_loss += [loss.item()]

        self.pae.train()
        self.dataset.set_mode("train")

        return np.mean(total_loss)

    def train(self, window_size: int) -> DataLoader:
        ckpt_epoch = self.ckpt_period
        best_model_state = None

        best_val_loss = np.inf
        patience_counter = 0

        for epoch in range(self.n_epochs):
            error = 0.0
            self.pae.scheduler.step()

            for train_batch in self.data_loader:
                reconstructed, _ = self.pae(train_batch)

                loss = self.pae.loss_fn(reconstructed, train_batch)

                self.pae.optimizer.zero_grad()
                loss.backward()
                self.pae.optimizer.step()
                self.pae.scheduler.batch_step()

                error += loss.item()

            avg_train_loss = error / len(self.data_loader)
            avg_val_loss = self._validate()

            if avg_val_loss < (best_val_loss - self.min_delta):
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.pae.state_dict().copy()
            else:
                patience_counter += 1

            wandb.log(
                {
                    "Reconstruction Loss (Train)": avg_train_loss,
                    "Reconstruction Loss (Val)": avg_val_loss,
                    "Early Stopping Counter": patience_counter,
                }
            )

            print(
                f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, "
                f"Validation Loss: {avg_val_loss:.4f}, "
                f"Early Stopping Counter: {patience_counter}"
            )

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                if best_model_state is not None:
                    self.pae.load_state_dict(best_model_state)
                    self.save_model()
                break

            # save and plot results at same frequency as cyclic scheduler
            if epoch != 0 and epoch % (ckpt_epoch - 1) == 0:
                self.save_model(epoch + 1)
                if self.input_channels == self.n_joints:
                    self.plot_joint_vels(window_size, epoch + 1)
                else:
                    self.plot_joint_states(window_size, epoch + 1)
                self.ckpt_period *= 2
                ckpt_epoch += self.ckpt_period

        wandb.finish()

    def save_model(self, epoch: int = None) -> np.ndarray:
        parameters = None
        self.pae.eval()
        self.dataset.set_mode("total")

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        if epoch is None:
            print("Saving Parameters")
            param_path = f"{self.ckptpath}/{self.version_no}_parameters_{self.phase_channels}phases_{self.intermediate_channels}intermediate_{self.frames}frames_{self.full_joint_state}.txt"
            with open(param_path, "w") as file:
                for i, batch in enumerate(dataloader):
                    with torch.no_grad():
                        _, params = self.pae(batch)
                    shift = helper.to_numpy(params[0]).squeeze()
                    freq = helper.to_numpy(params[1]).squeeze()
                    amp = helper.to_numpy(params[2]).squeeze()
                    offset = helper.to_numpy(params[3]).squeeze()

                    for j in range(shift.shape[0]):
                        param_values = np.concatenate(
                            (
                                shift[j, :],
                                freq[j, :],
                                amp[j, :],
                                offset[j],
                            )
                        )
                        line = " ".join(map(str, param_values))
                        file.write(line + "\n")
            parameters = np.loadtxt(param_path, dtype=np.float32)

        if epoch is not None:
            ckpt_name = f"{self.ckptpath}/pae_{self.version_no}_{self.phase_channels}phases_{self.intermediate_channels}intermediate_{self.frames}frames_{self.full_joint_state}_{epoch}.pt"
        else:
            ckpt_name = f"{self.ckptpath}/pae_{self.version_no}_{self.phase_channels}phases_{self.intermediate_channels}intermediate_{self.frames}frames_{self.full_joint_state}.pt"
        torch.save(self.pae.state_dict(), ckpt_name)

        return parameters

    def plot_joint_vels(self, window_size: int, epoch: int = None) -> None:
        self.pae.eval()
        self.dataset.set_mode("eval")
        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        original_history = []
        reconstructed_history = []

        current_frame = self.frames // 2

        for i, data in enumerate(dataloader):
            if i == window_size:
                break

            with torch.no_grad():
                reconstructed, _ = self.pae(data)

            original = data[..., current_frame].squeeze(0)
            reconstructed = reconstructed[..., current_frame].squeeze(0)

            original_history.append(helper.item(original))
            reconstructed_history.append(helper.item(reconstructed))

        original_history = torch.stack(original_history)
        reconstructed_history = torch.stack(reconstructed_history)

        fig, axes = plt.subplots(6, 4, figsize=(20, 15))
        axes = axes.flatten()
        for joint in range(self.n_joints):
            plotter.plot_continuous(
                axes[joint],
                original_history[:, joint],
                reconstructed_history[:, joint],
                title=f"Joint {joint}",
                xlabel="Frame",
                ylabel="Velocity",
            )
        plotter.save_img(
            fig, axes, self.n_joints, self.imgpath, filename=f"{epoch}_joint_velocities"
        )

        self.pae.train()
        self.dataset.set_mode("train")

    def plot_joint_states(self, window_size: int, epoch: int = None) -> None:
        self.pae.eval()
        self.dataset.set_mode("eval")
        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        current_frame = self.frames // 2

        original_positions_history = []
        original_velocities_history = []
        reconstructed_positions_history = []
        reconstructed_velocities_history = []

        for i, original in enumerate(dataloader):
            if i == window_size:
                break

            with torch.no_grad():
                reconstructed, _ = self.pae(original)

            original = original[..., current_frame].squeeze(0)
            reconstructed = reconstructed[..., current_frame].squeeze(0)

            original_positions = original[: self.n_joints]
            original_velocities = original[self.n_joints :]
            reconstructed_positions = reconstructed[: self.n_joints]
            reconstructed_velocities = reconstructed[self.n_joints :]

            original_positions_history.append(helper.item(original_positions))
            original_velocities_history.append(helper.item(original_velocities))
            reconstructed_positions_history.append(helper.item(reconstructed_positions))
            reconstructed_velocities_history.append(
                helper.item(reconstructed_velocities)
            )

        original_positions_history = torch.stack(original_positions_history)
        original_velocities_history = torch.stack(original_velocities_history)
        reconstructed_positions_history = torch.stack(reconstructed_positions_history)
        reconstructed_velocities_history = torch.stack(reconstructed_velocities_history)

        fig, axes = plt.subplots(6, 4, figsize=(20, 15))
        axes = axes.flatten()
        for joint in range(self.n_joints):
            plotter.plot_continuous(
                axes[joint],
                original_positions_history[:, joint],
                reconstructed_positions_history[:, joint],
                title=f"Joint {joint}",
                xlabel="Frame",
                ylabel="Position",
            )
        plotter.save_img(
            fig, axes, self.n_joints, self.imgpath, filename=f"{epoch}_joint_positions"
        )

        fig, axes = plt.subplots(6, 4, figsize=(20, 15))
        axes = axes.flatten()
        for joint in range(self.n_joints):
            plotter.plot_continuous(
                axes[joint],
                original_velocities_history[:, joint],
                reconstructed_velocities_history[:, joint],
                title=f"Joint {joint}",
                xlabel="Frame",
                ylabel="Velocity",
            )
        plotter.save_img(
            fig, axes, self.n_joints, self.imgpath, filename=f"{epoch}_joint_velocities"
        )

        self.pae.train()
        self.dataset.set_mode("train")


def plot_joint_gather_windows(
    pae: nn.Module, dataset: DatasetPAE, n_joints: int, frames: int
) -> None:
    pae.eval()
    dataset.set_mode("eval")

    for i in range(0, len(dataset), frames):
        original = dataset[i].unsqueeze(0)

        with torch.no_grad():
            reconstructed, _ = pae(original)

        original = original.squeeze(0)
        reconstructed = reconstructed.squeeze(0)

        fig, axes = plt.subplots(6, 4, figsize=(20, 15))
        axes = axes.flatten()
        for joint in range(n_joints):
            plotter.plot_continuous(
                axes[joint],
                original[joint, :],
                reconstructed[joint, :],
                title=f"Joint {joint}",
                # xlabel="Gather Window",
                ylabel="Velocity",
            )
        plt.tight_layout()
        plt.show()


def plot_phases(
    parameters: np.array, n_channels: int, frames: int, window_size: int
) -> None:
    shift = parameters[:, :n_channels]
    amp = parameters[:, 2 * n_channels : 3 * n_channels]

    phase_x = np.cos(2 * np.pi * shift[frames : window_size + frames])
    phase_y = np.sin(2 * np.pi * shift[frames : window_size + frames])

    scaled_phase_x = amp[frames : window_size + frames] * phase_x
    scaled_phase_y = amp[frames : window_size + frames] * phase_y

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=100)
    axes = axes.flatten()

    raw_color = "viridis"
    scaled_color = "plasma"

    for i in range(n_channels):
        scatter_raw = axes[i].scatter(
            phase_x[:, i],
            phase_y[:, i],
            c=np.arange(window_size),
            cmap=raw_color,
            s=3,
            alpha=0.6,
            label="Raw Phase",
        )
        scatter_scaled = axes[i].scatter(
            scaled_phase_x[:, i],
            scaled_phase_y[:, i],
            c=np.arange(window_size),
            cmap=scaled_color,
            s=3,
            alpha=0.6,
            label="Amplitude-Scaled Phase",
        )

        axes[i].set_title(f"Phase Channel {i+1}", fontsize=12, pad=10)
        axes[i].set_xlabel("Phase X", fontsize=10)
        axes[i].set_ylabel("Phase Y", fontsize=10)
        axes[i].grid(True, alpha=0.2, linestyle=":")
        axes[i].set_aspect("equal")

        min_amp = amp[:window_size, i].min()
        max_amp = amp[:window_size, i].max()
        axes[i].text(
            0.02,
            0.98,
            f"Max Amp: {max_amp:.2f}\nMin Amp: {min_amp:.2f}",
            transform=axes[i].transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        axes[i].axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        axes[i].axvline(x=0, color="gray", linestyle="-", alpha=0.3)

    cbar_ax1 = fig.add_axes([0.92, 0.55, 0.02, 0.3])
    cbar1 = plt.colorbar(scatter_raw, cax=cbar_ax1)
    cbar1.set_label("Time Step (Raw)", fontsize=10)

    cbar_ax2 = fig.add_axes([0.92, 0.15, 0.02, 0.3])
    cbar2 = plt.colorbar(scatter_scaled, cax=cbar_ax2)
    cbar2.set_label("Time Step (Scaled)", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()


def plot_frequencies(
    parameters: np.array, n_channels: int, frames: int, window_size: int
) -> None:
    freq = parameters[:, n_channels : 2 * n_channels]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=100)
    axes = axes.flatten()

    time_steps = np.arange(window_size)

    for i in range(n_channels):
        axes[i].plot(
            time_steps,
            freq[frames : window_size + frames, i],
            linewidth=2,
            color="blue",
            alpha=0.8,
        )

        axes[i].set_title(f"Frequency Channel {i+1}", fontsize=12, pad=10)
        axes[i].set_xlabel("Time Steps", fontsize=10)
        axes[i].set_ylabel("Frequency", fontsize=10)
        axes[i].grid(True, alpha=0.2, linestyle=":")

        min_freq = freq[:window_size, i].min()
        max_freq = freq[:window_size, i].max()
        axes[i].text(
            0.02,
            0.98,
            f"Max: {max_freq:.2f}\nMin: {min_freq:.2f}",
            transform=axes[i].transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_amplitudes(
    parameters: np.array, n_channels: int, frames: int, window_size: int
) -> None:
    amp = parameters[:, 2 * n_channels : 3 * n_channels]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=100)
    axes = axes.flatten()

    time_steps = np.arange(window_size)

    for i in range(n_channels):
        axes[i].plot(
            time_steps,
            amp[frames : window_size + frames, i],
            linewidth=2,
            color="purple",
            alpha=0.8,
        )

        axes[i].set_title(f"Amplitude Channel {i+1}", fontsize=12, pad=10)
        axes[i].set_xlabel("Time Steps", fontsize=10)
        axes[i].set_ylabel("Amplitude", fontsize=10)
        axes[i].grid(True, alpha=0.2, linestyle=":")

        min_amp = amp[:window_size, i].min()
        max_amp = amp[:window_size, i].max()
        axes[i].text(
            0.02,
            0.98,
            f"Max: {max_amp:.2f}\nMin: {min_amp:.2f}",
            transform=axes[i].transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    num_images = 10
    window_size = 600
    load_model = True

    trainer = TrainerPAE()

    if load_model:
        parameters = trainer.load()
    else:
        print("Model not loaded. Training PAE from scratch.")
        trainer.init_wandb()
        trainer.train(window_size)
        parameters = trainer.save_model()

    frames = trainer.frames
    n_channels = trainer.phase_channels

    if trainer.full_joint_state:
        trainer.plot_joint_states(window_size)
    else:
        trainer.plot_joint_vels(window_size)

    # plot_joint_gather_windows(
    #     trainer.pae, trainer.dataset, trainer.n_joints, trainer.frames
    # )
    plot_phases(parameters, n_channels, frames, window_size)
    plot_frequencies(parameters, n_channels, frames, window_size)
    plot_amplitudes(parameters, n_channels, frames, window_size)

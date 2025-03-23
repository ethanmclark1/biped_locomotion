import os
import yaml
import torch
import wandb

import torch.nn.functional as F
import biped_locomotion.scripts.utils.helper as helper
import biped_locomotion.scripts.utils.plotter as plotter

from torch.utils.data import DataLoader

from biped_locomotion.scripts.mann.network import MANN
from biped_locomotion.scripts.mann.dataset import DatasetMANN, TARGET_STRUCTURE


class TrainerMANN:
    def __init__(self) -> None:
        self.config = None
        config_path = "biped_locomotion/scripts/config.yaml"

        self._init_params(config_path)

        self.dataset = DatasetMANN(
            self.seed,
            self.n_joints,
            self.intermediate_channels,
            self.phase_channels,
            self.total_frames,
            self.sampled_frames,
            self.validation_ratio,
            self.datapath,
            self.ckptpath,
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

        self.mann = helper.to_device(
            MANN(
                self.dataset.x_norm,
                self.dataset.y_norm,
                self.batch_size,
                self.n_samples,
                self.config,
            )
        )

    def _init_params(self, config_path: str) -> None:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        fps = config["fps"]
        self.window = config["window"]
        self.total_frames = int(fps * self.window) + 1
        self.sampled_frames = config["MANN"]["sampled_frames"]
        self.n_joints = config["n_joints"]

        self.intermediate_channels = config["intermediate_channels"]
        self.phase_channels = config["phase_channels"]

        self.datapath = config["datapath"]
        self.ckptpath = config["ckptpath"]
        imgpath = config["imgpath"]
        imgpath = os.path.join(imgpath, "mann")

        self.ckpt_period = config["ckpt_period"]
        self.validation_ratio = config["validation_ratio"]

        self.n_epochs = config["MANN"]["n_epochs"]
        self.batch_size = config["MANN"]["batch_size"]
        self.full_joint_state = config["PAE"]["full_joint_state"]

        self.seed = config["seed"]
        helper.set_seed(self.seed)

        self.imgpath = os.path.join(
            imgpath,
            f"{self.phase_channels}phases_{self.intermediate_channels}intermediate_{self.total_frames}frames_{self.full_joint_state}",
        )
        os.makedirs(self.imgpath, exist_ok=True)
        os.makedirs(self.ckptpath, exist_ok=True)

        self.config = config

    def load(self) -> bool:
        loaded_model = False

        model_path = f"{self.ckptpath}/mann_{self.phase_channels}phases_{self.intermediate_channels}intermediate_{self.frames}frames_{self.full_joint_state}.pt"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.mann.load_state_dict(checkpoint)
            loaded_model = True
            print("Model loaded from", model_path)
        else:
            print("Model not found at", model_path)
            raise FileNotFoundError("Model not found at", model_path)

        return loaded_model

    def init_wandb(self) -> None:
        wandb.init(
            project="Kinematic Predictor",
            entity="ethanmclark1",
            name="Mode-Adaptive Neural Network",
        )

        for key, value in self.config.items():
            if isinstance(value, dict):
                if key == "PAE":
                    continue

                for k, v in value.items():
                    wandb.config[k] = v
            else:
                wandb.config[key] = value

        wandb.config["frames"] = self.total_frames

    def _validate(self) -> dict:
        self.mann.eval()
        self.dataset.set_mode("eval")

        val_loss = {
            "robot_projected_gravity_loss": 0.0,
            "robot_oriented_xy_vel_loss": 0.0,
            "robot_oriented_yaw_vel_loss": 0.0,
            "joint_pos_loss": 0.0,
            "joint_vel_loss": 0.0,
            "foot_contacts_loss": 0.0,
            "phase_update_loss": 0.0,
            "total_loss": 0.0,
        }
        for x_batch, y_batch in self.data_loader:
            with torch.no_grad():
                y_pred = self.mann(x_batch)

            normed_y_pred = helper.normalize(y_pred, self.mann.y_norm)
            normed_y_batch = helper.normalize(y_batch, self.mann.y_norm)
            loss = self.mann.loss_fn(normed_y_pred, normed_y_batch)

            val_loss["robot_projected_gravity_loss"] += F.mse_loss(
                normed_y_pred[:, TARGET_STRUCTURE["robot_projected_gravity"]],
                normed_y_batch[:, TARGET_STRUCTURE["robot_projected_gravity"]],
            )
            val_loss["robot_oriented_xy_vel_loss"] += F.mse_loss(
                normed_y_pred[:, TARGET_STRUCTURE["robot_oriented_xy_vel"]],
                normed_y_batch[:, TARGET_STRUCTURE["robot_oriented_xy_vel"]],
            )
            val_loss["robot_oriented_yaw_vel_loss"] += F.mse_loss(
                normed_y_pred[:, TARGET_STRUCTURE["robot_oriented_yaw_vel"]],
                normed_y_batch[:, TARGET_STRUCTURE["robot_oriented_yaw_vel"]],
            )
            val_loss["joint_pos_loss"] += F.mse_loss(
                normed_y_pred[:, TARGET_STRUCTURE["joint_positions"]],
                normed_y_batch[:, TARGET_STRUCTURE["joint_positions"]],
            )
            val_loss["joint_vel_loss"] += F.mse_loss(
                normed_y_pred[:, TARGET_STRUCTURE["joint_velocities"]],
                normed_y_batch[:, TARGET_STRUCTURE["joint_velocities"]],
            )
            val_loss["foot_contacts_loss"] += F.binary_cross_entropy_with_logits(
                y_pred[:, TARGET_STRUCTURE["foot_contacts"]],
                y_batch[:, TARGET_STRUCTURE["foot_contacts"]],
            )
            val_loss["phase_update_loss"] += F.mse_loss(
                normed_y_pred[:, TARGET_STRUCTURE["phase_update"]],
                normed_y_batch[:, TARGET_STRUCTURE["phase_update"]],
            )
            val_loss["total_loss"] += loss.item()

        val_loss = {k: v / len(self.data_loader) for k, v in val_loss.items()}

        self.mann.train()
        self.dataset.set_mode("train")

        return val_loss

    def train(self, window_size: int) -> None:
        ckpt_epoch = self.ckpt_period
        for epoch in range(self.n_epochs):
            train_loss = {
                "robot_projected_gravity_loss": 0.0,
                "robot_oriented_xy_vel_loss": 0.0,
                "robot_oriented_yaw_vel_loss": 0.0,
                "joint_pos_loss": 0.0,
                "joint_vel_loss": 0.0,
                "foot_contacts_loss": 0.0,
                "phase_update_loss": 0.0,
                "total_loss": 0.0,
            }
            self.mann.scheduler.step()
            for x_batch, y_batch in self.data_loader:
                assert self.mann.training and self.dataset.mode == "train"

                y_pred = self.mann(x_batch)

                normed_y_pred = helper.normalize(y_pred, self.mann.y_norm)
                normed_y_batch = helper.normalize(y_batch, self.mann.y_norm)
                loss = self.mann.loss_fn(normed_y_pred, normed_y_batch)

                self.mann.optimizer.zero_grad()
                loss.backward()
                self.mann.optimizer.step()
                self.mann.scheduler.batch_step()

                train_loss["robot_projected_gravity_loss"] += F.mse_loss(
                    normed_y_pred[:, TARGET_STRUCTURE["robot_projected_gravity"]],
                    normed_y_batch[:, TARGET_STRUCTURE["robot_projected_gravity"]],
                )
                train_loss["robot_oriented_xy_vel_loss"] += F.mse_loss(
                    normed_y_pred[:, TARGET_STRUCTURE["robot_oriented_xy_vel"]],
                    normed_y_batch[:, TARGET_STRUCTURE["robot_oriented_xy_vel"]],
                )
                train_loss["robot_oriented_yaw_vel_loss"] += F.mse_loss(
                    normed_y_pred[:, TARGET_STRUCTURE["robot_oriented_yaw_vel"]],
                    normed_y_batch[:, TARGET_STRUCTURE["robot_oriented_yaw_vel"]],
                )
                train_loss["joint_pos_loss"] += F.mse_loss(
                    normed_y_pred[:, TARGET_STRUCTURE["joint_positions"]],
                    normed_y_batch[:, TARGET_STRUCTURE["joint_positions"]],
                )
                train_loss["joint_vel_loss"] += F.mse_loss(
                    normed_y_pred[:, TARGET_STRUCTURE["joint_velocities"]],
                    normed_y_batch[:, TARGET_STRUCTURE["joint_velocities"]],
                )
                train_loss["foot_contacts_loss"] += F.binary_cross_entropy_with_logits(
                    y_pred[:, TARGET_STRUCTURE["foot_contacts"]],
                    y_batch[:, TARGET_STRUCTURE["foot_contacts"]],
                )
                train_loss["phase_update_loss"] += F.mse_loss(
                    normed_y_pred[:, TARGET_STRUCTURE["phase_update"]],
                    normed_y_batch[:, TARGET_STRUCTURE["phase_update"]],
                )
                train_loss["total_loss"] += loss.item()

            avg_train_loss = {
                k: v / len(self.data_loader) for k, v in train_loss.items()
            }
            avg_val_loss = self._validate()

            wandb.log(
                {
                    "Root Projected Gravity Loss (Train)": avg_train_loss[
                        "robot_projected_gravity_loss"
                    ],
                    "Root Projected Gravity Loss (Val)": avg_val_loss[
                        "robot_projected_gravity_loss"
                    ],
                    "World-Oriented Linear Velocity Loss (Train)": avg_train_loss[
                        "robot_oriented_xy_vel_loss"
                    ],
                    "World-Oriented Linear Velocity Loss (Val)": avg_val_loss[
                        "robot_oriented_xy_vel_loss"
                    ],
                    "World-Oriented Yaw Velocity Loss (Train)": avg_train_loss[
                        "robot_oriented_yaw_vel_loss"
                    ],
                    "World-Oriented Yaw Velocity Loss (Val)": avg_val_loss[
                        "robot_oriented_yaw_vel_loss"
                    ],
                    "Joint Position Loss (Train)": avg_train_loss["joint_pos_loss"],
                    "Joint Position Loss (Val)": avg_val_loss["joint_pos_loss"],
                    "Joint Velocity Loss (Train)": avg_train_loss["joint_vel_loss"],
                    "Joint Velocity Loss (Val)": avg_val_loss["joint_vel_loss"],
                    "Foot Contacts Loss (Train)": avg_train_loss["foot_contacts_loss"],
                    "Foot Contacts Loss (Val)": avg_val_loss["foot_contacts_loss"],
                    "Phase Update Loss (Train)": avg_train_loss["phase_update_loss"],
                    "Phase Update Loss (Val)": avg_val_loss["phase_update_loss"],
                    "Total Loss (Train)": avg_train_loss["total_loss"],
                    "Total Loss (Val)": avg_val_loss["total_loss"],
                }
            )

            print(
                f"Epoch {epoch + 1}, Train Loss: {avg_train_loss['total_loss']:.4f}, Validation Loss: {avg_val_loss['total_loss']:.4f}"
            )

            # save and plot results at same frequency as cyclic scheduler
            if epoch != 0 and epoch % (ckpt_epoch - 1) == 0:
                self.save_model(epoch + 1)
                self.plot_results(window_size, epoch + 1)
                self.ckpt_period *= 2
                ckpt_epoch += self.ckpt_period

        wandb.finish()

    def save_model(self, epoch: int = None) -> None:
        if epoch is not None:
            ckpt_name = f"{self.ckptpath}/mann_{self.phase_channels}phases_{self.intermediate_channels}intermediate_{self.total_frames}frames_{self.full_joint_state}_epoch{epoch}.pt"
        else:
            ckpt_name = f"{self.ckptpath}/mann_{self.phase_channels}phases_{self.intermediate_channels}intermediate_{self.total_frames}frames_{self.full_joint_state}.pt"

        torch.save(self.mann.state_dict(), ckpt_name)

        print("Model saved to", ckpt_name)

    def plot_results(
        self,
        window_size: int,
        epoch: int = None,
    ) -> None:

        self.mann.eval()
        self.dataset.set_mode("eval")
        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        predicted_robot_projected_gravity = []
        predicted_robot_oriented_xy_vel = []
        predicted_robot_oriented_yaw_vel = []
        predicted_joint_positions = []
        predicted_joint_velocities = []
        predicted_foot_contacts = []
        predicted_phase_updates = []

        target_robot_projected_gravity = []
        target_robot_oriented_xy_vel = []
        target_robot_oriented_yaw_vel = []
        target_joint_positions = []
        target_joint_velocities = []
        target_foot_contacts = []
        target_phase_updates = []

        for i, (input, target) in enumerate(dataloader):
            if i == window_size:
                break

            with torch.no_grad():
                y_pred = self.mann(input)

            predicted_robot_projected_gravity.append(
                helper.item(y_pred[0, TARGET_STRUCTURE["robot_projected_gravity"]])
            )
            predicted_robot_oriented_xy_vel.append(
                helper.item(y_pred[0, TARGET_STRUCTURE["robot_oriented_xy_vel"]])
            )
            predicted_robot_oriented_yaw_vel.append(
                helper.item(y_pred[0, TARGET_STRUCTURE["robot_oriented_yaw_vel"]])
            )
            predicted_joint_positions.append(
                helper.item(y_pred[0, TARGET_STRUCTURE["joint_positions"]])
            )
            predicted_joint_velocities.append(
                helper.item(y_pred[0, TARGET_STRUCTURE["joint_velocities"]])
            )
            predicted_foot_contacts.append(
                helper.item(y_pred[0, TARGET_STRUCTURE["foot_contacts"]])
            )
            predicted_phase_updates.append(
                helper.item(y_pred[0, TARGET_STRUCTURE["phase_update"]])
            )

            target_robot_projected_gravity.append(
                helper.item(target[0, TARGET_STRUCTURE["robot_projected_gravity"]])
            )
            target_robot_oriented_xy_vel.append(
                helper.item(target[0, TARGET_STRUCTURE["robot_oriented_xy_vel"]])
            )
            target_robot_oriented_yaw_vel.append(
                helper.item(target[0, TARGET_STRUCTURE["robot_oriented_yaw_vel"]])
            )
            target_joint_positions.append(
                helper.item(target[0, TARGET_STRUCTURE["joint_positions"]])
            )
            target_joint_velocities.append(
                helper.item(target[0, TARGET_STRUCTURE["joint_velocities"]])
            )
            target_foot_contacts.append(
                helper.item(target[0, TARGET_STRUCTURE["foot_contacts"]])
            )
            target_phase_updates.append(
                helper.item(target[0, TARGET_STRUCTURE["phase_update"]])
            )

        predicted_robot_projected_gravity = torch.stack(
            predicted_robot_projected_gravity
        )
        predicted_robot_oriented_xy_vel = torch.stack(predicted_robot_oriented_xy_vel)
        predicted_robot_oriented_yaw_vel = torch.stack(predicted_robot_oriented_yaw_vel)
        predicted_joint_positions = torch.stack(predicted_joint_positions)
        predicted_joint_velocities = torch.stack(predicted_joint_velocities)
        predicted_foot_contacts = torch.stack(predicted_foot_contacts)
        predicted_phase_updates = torch.stack(predicted_phase_updates)

        target_robot_projected_gravity = torch.stack(target_robot_projected_gravity)
        target_robot_oriented_xy_vel = torch.stack(target_robot_oriented_xy_vel)
        target_robot_oriented_yaw_vel = torch.stack(target_robot_oriented_yaw_vel)
        target_joint_positions = torch.stack(target_joint_positions)
        target_joint_velocities = torch.stack(target_joint_velocities)
        target_foot_contacts = torch.stack(target_foot_contacts)
        target_phase_updates = torch.stack(target_phase_updates)

        plotter.plot_results(
            predicted_robot_projected_gravity=predicted_robot_projected_gravity,
            predicted_robot_oriented_xy_vel=predicted_robot_oriented_xy_vel,
            predicted_robot_oriented_yaw_vel=predicted_robot_oriented_yaw_vel,
            predicted_joint_positions=predicted_joint_positions,
            predicted_joint_velocities=predicted_joint_velocities,
            predicted_foot_contacts=predicted_foot_contacts,
            predicted_phase_updates=predicted_phase_updates,
            target_robot_projected_gravity=target_robot_projected_gravity,
            target_robot_oriented_xy_vel=target_robot_oriented_xy_vel,
            target_robot_oriented_yaw_vel=target_robot_oriented_yaw_vel,
            target_joint_positions=target_joint_positions,
            target_joint_velocities=target_joint_velocities,
            target_foot_contacts=target_foot_contacts,
            target_phase_updates=target_phase_updates,
            imgpath=self.imgpath,
            counter=epoch,
            n_joints=self.n_joints,
            phase_channels=self.phase_channels,
            sampled_frames=self.sampled_frames,
        )

        self.mann.train()
        self.dataset.set_mode("train")


if __name__ == "__main__":
    window_size = 300
    load_model = False

    trainer = TrainerMANN()

    if load_model:
        trainer.load()
    else:
        print("Model not loaded. Training MANN from scratch.")
        trainer.init_wandb()
        trainer.train(window_size)
        trainer.save_model()

    trainer.plot_results(window_size)

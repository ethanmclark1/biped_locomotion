import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import humanoid_loco.scripts.utils.helper as helper

from humanoid_loco.scripts.utils.adam_wr.adam_w import AdamW
from humanoid_loco.scripts.utils.adam_wr.cyclic_scheduler import (
    CyclicLRWithRestarts,
)

from humanoid_loco.scripts.mann.dataset import INPUT_STRUCTURE, TARGET_STRUCTURE


# Output-Blended MoE Layer
class ExpertsLinear(nn.Module):
    def __init__(self, experts: int, input_dim: int, output_dim: int) -> None:
        super(ExpertsLinear, self).__init__()

        self.experts = experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = self.weights([experts, input_dim, output_dim])
        self.b = self.bias([experts, 1, output_dim])

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        y = torch.zeros(
            (x.shape[0], self.output_dim), device=x.device, requires_grad=True
        )
        for i in range(self.experts):
            y = y + weights[:, i].unsqueeze(1) * (
                x.matmul(self.W[i, :, :]) + self.b[i, :, :]
            )
        return y

    def weights(self, shape: list) -> nn.Parameter:
        alpha_bound = np.sqrt(6.0 / np.prod(shape[-2:]))
        alpha = np.asarray(
            np.random.uniform(low=-alpha_bound, high=alpha_bound, size=shape),
            dtype=np.float32,
        )
        return nn.Parameter(torch.from_numpy(alpha), requires_grad=True)

    def bias(self, shape: list) -> nn.Parameter:
        return nn.Parameter(torch.zeros(shape, dtype=torch.float32), requires_grad=True)


class MANN(nn.Module):
    def __init__(
        self,
        input_norm: torch.Tensor,
        output_norm: torch.Tensor,
        batch_size: int,
        n_samples: int,
        config: dict,
    ) -> None:

        super(MANN, self).__init__()

        self.name = self.__class__.__name__

        n_experts = config[self.name]["n_experts"]
        main_hidden = config[self.name]["main_hidden"]
        gating_hidden = config[self.name]["gating_hidden"]

        t_mult = config[self.name]["t_mult"]
        dropout = config[self.name]["dropout"]
        weight_decay = config[self.name]["weight_decay"]
        learning_rate = config[self.name]["learning_rate"]
        restart_period = config[self.name]["restart_period"]

        input_components = list(INPUT_STRUCTURE.keys())
        last_character_component = input_components[-2]
        phase_component = input_components[-1]
        n_character_indices = INPUT_STRUCTURE[last_character_component].stop
        n_gating_indices = INPUT_STRUCTURE[phase_component].stop - n_character_indices

        output_dims = list(TARGET_STRUCTURE.values())[-1].stop

        self.character_indices = torch.arange(n_character_indices)
        self.gating_indices = torch.arange(
            n_character_indices, n_character_indices + n_gating_indices
        )

        if n_gating_indices + n_character_indices != input_norm.shape[1]:
            print(
                f"Number of gating features {n_gating_indices} and character features {n_character_indices} are not the same as input features {input_norm.shape[1]}"
            )
            raise ValueError("Number of features do not match")

        self.gating_0 = nn.Linear(n_gating_indices, gating_hidden)
        self.gating_1 = nn.Linear(gating_hidden, gating_hidden)
        self.gating_2 = nn.Linear(gating_hidden, n_experts)

        self.experts_0 = ExpertsLinear(n_experts, n_character_indices, main_hidden)
        self.experts_1 = ExpertsLinear(n_experts, main_hidden, main_hidden)
        self.experts_2 = ExpertsLinear(n_experts, main_hidden, output_dims)

        self.dropout = dropout
        self.x_norm = nn.Parameter(input_norm, requires_grad=False)
        self.y_norm = nn.Parameter(output_norm, requires_grad=False)

        self.loss_fn = torch.nn.MSELoss()

        self.optimizer = AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.scheduler = CyclicLRWithRestarts(
            self.optimizer,
            batch_size,
            epoch_size=n_samples,
            restart_period=restart_period,
            t_mult=t_mult,
            policy="cosine",
            verbose=True,
        )

    def forward(self, combined_input: torch.Tensor) -> torch.Tensor:
        combined_input = helper.normalize(combined_input, self.x_norm)

        # Gating Network
        phase_input = combined_input[:, self.gating_indices]

        phase = F.dropout(phase_input, self.dropout, training=self.training)
        phase = self.gating_0(phase)
        phase = F.elu(phase)

        phase = F.dropout(phase, self.dropout, training=self.training)
        phase = self.gating_1(phase)
        phase = F.elu(phase)

        phase = F.dropout(phase, self.dropout, training=self.training)
        phase = self.gating_2(phase)

        expert_weights = F.softmax(phase, dim=1)

        # Motion Prediction Network
        character_input = combined_input[:, self.character_indices]

        character = F.dropout(character_input, self.dropout, training=self.training)
        character = self.experts_0(character, expert_weights)
        character = F.elu(character)

        character = F.dropout(character, self.dropout, training=self.training)
        character = self.experts_1(character, expert_weights)
        character = F.elu(character)

        character = F.dropout(character, self.dropout, training=self.training)
        character = self.experts_2(character, expert_weights)

        return helper.renormalize(character, self.y_norm)

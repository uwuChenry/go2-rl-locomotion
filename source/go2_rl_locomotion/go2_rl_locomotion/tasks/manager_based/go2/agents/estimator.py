# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Concurrent state estimation for RSL-RL, as used by Walk These Ways.

WTW (Margolis & Agrawal, CoRL 2022, Sec. 3.2) feeds the policy "estimated domain parameters: the velocity of the
robot body and the ground friction, which are predicted from the observation history using supervised learning in
the manner of [Ji et al. 2022]". The estimator is an MLP with hidden layers [256, 128] and ELU activations.

Two pieces implement that on top of RSL-RL 5:

* :class:`EstimatorMLPModel` -- an actor whose input is the policy observation *plus* the estimator's prediction.
  The prediction is detached, so the estimator is trained only by its own regression loss and PPO gradients never
  reach it. Exports (JIT/ONNX) run the estimator internally, so the deployed policy still takes just the policy
  observation.
* :class:`PPOWithEstimator` -- PPO plus a supervised MSE update of the estimator against a privileged observation
  group. It runs *after* the PPO update so that the action log-probs stored during the rollout stay consistent with
  the actor that PPO optimizes.

Only torch and rsl_rl are imported here, so this module is safe to resolve before ``SimulationApp`` starts.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, HiddenState


class EstimatorMLPModel(MLPModel):
    """MLP actor that appends a learned estimate of privileged state to its input.

    The estimator reads the same (normalized) observation as the policy body and predicts the observation group
    named ``estimator_target_group``. That group must be present in the environment observations, but it does not
    need to be (and normally is not) part of the actor's own observation set.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        estimator_target_group: str = "privileged",
        estimator_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        estimator_activation: str = "elu",
        **kwargs,
    ) -> None:
        # needed by _get_latent_dim(), which the base constructor calls
        self.estimate_dim = obs[estimator_target_group].shape[-1]
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        self.estimator_target_group = estimator_target_group
        self.estimator = MLP(self.obs_dim, self.estimate_dim, estimator_hidden_dims, estimator_activation)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        latent = super().get_latent(obs, masks, hidden_state)
        return torch.cat([latent, self.estimator(latent).detach()], dim=-1)

    def estimate(self, obs: TensorDict) -> torch.Tensor:
        """Predict the target group from the actor's observations."""
        return self.estimator(super().get_latent(obs))

    def estimation_loss(self, obs: TensorDict) -> torch.Tensor:
        """Mean squared error between the estimate and the true target group."""
        return (self.estimate(obs) - obs[self.estimator_target_group]).pow(2).mean()

    def as_jit(self) -> nn.Module:
        return _TorchEstimatorMLPModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxEstimatorMLPModel(self, verbose)

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.estimate_dim


class PPOWithEstimator(PPO):
    """PPO that also fits the actor's state estimator by supervised regression.

    The estimator gets its own Adam optimizer and the same epoch/mini-batch schedule as PPO, run over the rollout
    that PPO just consumed. ``RolloutStorage.clear()`` only rewinds the write cursor, so that data is still intact
    after :meth:`PPO.update` returns.
    """

    actor: EstimatorMLPModel

    def __init__(self, actor: EstimatorMLPModel, critic: MLPModel, storage, estimator_learning_rate: float = 1e-3, **kwargs):
        super().__init__(actor, critic, storage, **kwargs)
        if self.is_multi_gpu:
            raise NotImplementedError("PPOWithEstimator does not synchronize estimator gradients across GPUs.")
        if not isinstance(self.actor, EstimatorMLPModel):
            raise TypeError(f"PPOWithEstimator needs an EstimatorMLPModel actor, got {type(self.actor).__name__}.")
        self.estimator_optimizer = torch.optim.Adam(self.actor.estimator.parameters(), lr=estimator_learning_rate)

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        loss_dict["estimator"] = self._update_estimator()
        return loss_dict

    def _update_estimator(self) -> float:
        observations = self.storage.observations.flatten(0, 1)
        batch_size = observations.batch_size[0]
        mini_batch_size = batch_size // self.num_mini_batches
        mean_loss = 0.0
        for _ in range(self.num_learning_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for i in range(self.num_mini_batches):
                batch = observations[indices[i * mini_batch_size : (i + 1) * mini_batch_size]]
                loss = self.actor.estimation_loss(batch)
                self.estimator_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.estimator.parameters(), self.max_grad_norm)
                self.estimator_optimizer.step()
                mean_loss += loss.item()
        return mean_loss / (self.num_learning_epochs * self.num_mini_batches)

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["estimator_optimizer_state_dict"] = self.estimator_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if (load_cfg is None or load_cfg.get("optimizer")) and "estimator_optimizer_state_dict" in loaded_dict:
            self.estimator_optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
        return load_iteration


class _TorchEstimatorMLPModel(nn.Module):
    """Exportable estimator + policy for JIT. Takes only the policy observation."""

    def __init__(self, model: EstimatorMLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.estimator = copy.deepcopy(model.estimator)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = model.distribution.as_deterministic_output_module()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        x = torch.cat([x, self.estimator(x)], dim=-1)
        return self.deterministic_output(self.mlp(x))

    @torch.jit.export
    def estimate(self, x: torch.Tensor) -> torch.Tensor:
        """The estimator's prediction alone, for logging on the robot."""
        return self.estimator(self.obs_normalizer(x))

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxEstimatorMLPModel(_TorchEstimatorMLPModel):
    """Exportable estimator + policy for ONNX. Same input and output names as the stock MLP export."""

    is_recurrent: bool = False

    def __init__(self, model: EstimatorMLPModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]

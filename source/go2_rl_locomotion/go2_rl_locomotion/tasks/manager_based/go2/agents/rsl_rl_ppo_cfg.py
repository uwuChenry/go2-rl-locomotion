# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRNNModelCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 50
    experiment_name = "unitree_go2_flat"
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PPORunnerHistoryCfg(RslRlOnPolicyRunnerCfg):
    """PPO for the 10-frame observation-history environment.

    Same MLP architecture family as :class:`PPORunnerCfg`, but the input is 10x wider
    (48 obs per frame -> 480), so the first layer is widened to match. Everything else is
    kept identical to the baseline so the two runs stay comparable.
    """

    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 50
    experiment_name = "unitree_go2_flat_history"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RslRlEstimatorMLPModelCfg(RslRlMLPModelCfg):
    """MLP actor with a WTW-style state estimator on its input (see :mod:`.estimator`)."""

    class_name: str = "{DIR}.estimator:EstimatorMLPModel"

    estimator_target_group: str = "privileged"
    """Observation group the estimator learns to predict."""

    estimator_hidden_dims: list[int] = [256, 128]
    """Hidden layers of the estimator MLP. WTW uses [256, 128]."""

    estimator_activation: str = "elu"


@configclass
class RslRlPpoEstimatorAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO plus the supervised estimator update (see :mod:`.estimator`)."""

    class_name: str = "{DIR}.estimator:PPOWithEstimator"

    estimator_learning_rate: float = 1.0e-3


@configclass
class PPORunnerWTWCfg(RslRlOnPolicyRunnerCfg):
    """PPO for the Walk These Ways environment.

    The "privileged" group (true base linear velocity + foot friction) is used two ways,
    matching WTW Sec. 3.2:

    * the critic reads it directly -- asymmetric actor-critic;
    * the actor gets a *prediction* of it from an estimator MLP fed with the same 30-step
      history, trained by supervised regression alongside PPO. At deployment the estimator
      runs inside the exported policy, so the robot still only supplies the "policy" group.

    Actor input is 45 obs per frame x 30 frames = 1350, plus the 7-dim estimate (3 velocity
    + 4 foot frictions), so the network is widened accordingly.
    """

    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 50
    # new name: checkpoints from the estimator-less runs have a different actor and won't load
    experiment_name = "unitree_go2_flat_wtw_est"
    obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
    actor = RslRlEstimatorMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoEstimatorAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PPORunnerRNNCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 50
    experiment_name = "unitree_go2_flat_rnn"
    actor = RslRlRNNModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlRNNModelCfg.GaussianDistributionCfg(init_std=1.0),
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )
    critic = RslRlRNNModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PPORunnerRoughCfg(PPORunnerCfg):
    """Rough-terrain runner: wider MLPs to take in the 187-dim height scan."""

    experiment_name = "unitree_go2_rough"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )

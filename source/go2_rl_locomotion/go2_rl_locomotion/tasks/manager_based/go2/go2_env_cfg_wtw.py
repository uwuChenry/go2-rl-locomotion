# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 locomotion configured after "Walk These Ways" (Margolis & Agrawal, CoRL 2022).

Reference: https://arxiv.org/abs/2212.03238 (Unitree Go1, Isaac Gym, PPO)

This mirrors the paper's *sim-to-real recipe*, not its Multiplicity-of-Behavior gait
conditioning. Three things differ from :mod:`go2_env_cfg`:

1. **The actor never sees base linear velocity.** A real quadruped does not measure it --
   it comes out of a state estimator -- so WTW predicts it, together with ground friction,
   from observation history. Both live in a privileged group: the critic reads it directly
   (asymmetric actor-critic), and the actor gets a learned estimate of it from a supervised
   estimator network (see ``agents/estimator.py`` and :class:`PPORunnerWTWCfg`).
2. **30-step observation history** (the paper's value) instead of 10.
3. **Actuation latency and wider domain randomization**, matching the paper's Table 5.

Randomization ranges, side by side with WTW Table 5:

    parameter            WTW              here
    payload mass         -1.0 .. 3.0 kg   same
    motor strength       90 .. 110 %      same (as a PD-gain scale, see note below)
    joint calibration    -0.02 .. 0.02    same (as a joint_pos observation bias)
    ground friction      0.40 .. 1.00     same
    ground restitution   0.00 .. 1.00     same
    gravity offset       -1.0 .. 1.0      same (m/s^2, z axis)
    action latency       ~20 ms           1 control step = 20 ms at 50 Hz
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .delayed_actions_cfg import DelayedJointPositionActionCfg
from .go2_env_cfg import Go2EnvCfg
from . import observations as go2_obs

##
# Observations: policy (deployable) vs. privileged (critic only)
##


@configclass
class WTWObservationsCfg:
    """Observation specifications for the WTW-style environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Everything the real robot can actually measure.

        Note the absence of ``base_lin_vel``. What is left is exactly what a Go2 gives you:
        IMU angular rate, gravity direction from the accelerometer, joint encoders, and the
        policy's own previous action.
        """

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=Unoise(n_min=-0.15, n_max=0.15),
                bias_noise_cfg=Unoise(n_min=-0.025, n_max=0.025),
            ),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=Unoise(n_min=-0.04, n_max=0.04),
                bias_noise_cfg=Unoise(n_min=-0.0125, n_max=0.0125),
            ),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            # the bias here IS WTW's "joint calibration" randomization (-0.02 .. 0.02 rad):
            # a constant per-episode encoder offset, which is what miscalibration looks like
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=Unoise(n_min=-0.01, n_max=0.01),
                bias_noise_cfg=Unoise(n_min=-0.02, n_max=0.02),
            ),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=Unoise(n_min=-1.25, n_max=1.25),
                bias_noise_cfg=Unoise(n_min=-0.0625, n_max=0.0625),
            ),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            # WTW feeds a 30-step history of observations, commands and previous actions
            self.history_length = 30
            self.flatten_history_dim = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Ground truth the robot can never measure: WTW's estimated domain parameters.

        Used twice (see :class:`PPORunnerWTWCfg`): the critic reads it as input, and it is
        the regression target for the actor's estimator. The actor itself never reads it, so
        it never has to survive deployment.
        """

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        foot_friction = ObsTerm(
            func=go2_obs.foot_friction,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


##
# Actions: same joint-position control, with actuation latency
##


@configclass
class WTWActionsCfg:
    """Action specifications for the WTW-style environment."""

    joint_pos = DelayedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
        # WTW models a constant ~20 ms actuation delay; decimation 4 x sim.dt 0.005 = 50 Hz,
        # so one control step is exactly 20 ms
        delay_steps=(1, 1),
    )


##
# Events: WTW Table 5 randomization ranges
##


@configclass
class WTWEventCfg:
    """Domain randomization following WTW Table 5."""

    # -- startup

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.0),  # WTW: ground friction 0.40 - 1.00
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 1.0),  # WTW: ground restitution 0.00 - 1.00
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),  # WTW: payload mass -1.0 - 3.0 kg
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05)},
        },
    )

    # WTW's "motor strength 90 - 110 %". The paper scales commanded torque; the closest
    # supported knob here is the actuator PD gains, and for a PD-controlled joint
    # tau = kp * err, so scaling kp scales the delivered torque by the same factor.
    motor_strength = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    # -- reset

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # WTW: gravity offset -1.0 - 1.0 m/s^2. On the PhysX backend this is scene-wide rather
    # than per-environment, so it is resampled on an interval instead of at startup to keep
    # the policy from overfitting to one value.
    gravity_offset = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(8.0, 12.0),
        params={
            "gravity_distribution_params": ([0.0, 0.0, -1.0], [0.0, 0.0, 1.0]),
            "operation": "add",
        },
    )

    # -- interval

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


##
# Environment
##


@configclass
class Go2EnvCfgWTW(Go2EnvCfg):
    """Go2 velocity tracking with the Walk These Ways sim-to-real recipe.

    Rewards, terminations, commands, curriculum and the robot itself are inherited
    unchanged from :class:`Go2EnvCfg` -- only observations, actions and randomization
    differ.
    """

    observations: WTWObservationsCfg = WTWObservationsCfg()
    actions: WTWActionsCfg = WTWActionsCfg()
    events: WTWEventCfg = WTWEventCfg()


@configclass
class Go2EnvCfgWTW_PLAY(Go2EnvCfgWTW):
    """Play/test configuration for the WTW agent."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        # deploy-like conditions: no sensor corruption, no pushes, but keep the latency
        # since the real robot has it
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.gravity_offset = None

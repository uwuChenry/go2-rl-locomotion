# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .reward_functions import (
    FootClearanceQualityReward,
    FootStepQualityReward,
    feet_air_time_reward,
    feet_air_time_tent,
    feet_air_time_uncapped,
    foot_clearance_reward,
    foot_spread_reward,
    gait_reward,
    log_avg_feet_air_time,
    reward_base_height_above_threshold,
    reward_height_scan_above_threshold,
)

##
# Pre-defined configs
##
from .go2_articulation_cfg import GO2_BACK_PLATFORM_CFG


##
# Scene definition
##


@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a go2 robot."""

    # ground terrain - flat plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=PhysxRigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robot
    robot: ArticulationCfg = GO2_BACK_PLATFORM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=True,  # Enable ray visualization
    #     mesh_prim_paths=["/World/ground"],
    # )

    # Contact sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.65,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            # noise=NoiseModelWithAdditiveBiasCfg(
            #     noise_cfg=Unoise(n_min=-0.2, n_max=0.2),
            #     bias_noise_cfg=Unoise(n_min=-0.075, n_max=0.075),
            # ),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.15, n_max=0.15),
            # noise=NoiseModelWithAdditiveBiasCfg(
            #     noise_cfg=Unoise(n_min=-0.3, n_max=0.3),
            #     bias_noise_cfg=Unoise(n_min=-0.05, n_max=0.05),
            # ),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.04, n_max=0.04),
            # noise=NoiseModelWithAdditiveBiasCfg(
            #     noise_cfg=Unoise(n_min=-0.08, n_max=0.08),
            #     bias_noise_cfg=Unoise(n_min=-0.025, n_max=0.025),
            # ),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            # noise=NoiseModelWithAdditiveBiasCfg(
            #     noise_cfg=Unoise(n_min=-0.02, n_max=0.02),
            #     bias_noise_cfg=Unoise(n_min=-0.005, n_max=0.005),
            # ),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.25, n_max=1.25),
            # noise=NoiseModelWithAdditiveBiasCfg(
            #     noise_cfg=Unoise(n_min=-2.5, n_max=2.5),
            #     bias_noise_cfg=Unoise(n_min=-0.125, n_max=0.125),
            # ),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8), # 0.8 .0.8
            "dynamic_friction_range": (0.6, 0.6), #0.6 0.6
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
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

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5), #0.5 1.5
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.00001)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.6e-8)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # feet_air_time = RewTerm(
    #     func=feet_air_time_reward,
    #     weight=10, #0.125
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.02,   # 0 reward below this
    #         "cap_start": 0.3,   # ramps 0 -> 1 across [threshold, cap_start]
    #         "cap_end": 0.4,     # flat at 1 across [cap_start, cap_end]
    #         "hard_cap": 0.5,    # ramps 1 -> 0 across [cap_end, hard_cap]
    #         "command_deadzone": 0.0,  # no gate: reward stepping even at zero commanded velocity
    #     },
    # )

    feet_air_time2 = RewTerm(
        func=feet_air_time_tent,
        weight=20.0, #0.125
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 0.2, 
        },
    )

    foot_clearance_quality = RewTerm(
        func=FootClearanceQualityReward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 0.03,  # 0 reward for swings peaking below 3cm
            "cap": 0.1,         # full reward for swings peaking at/above 10cm
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "threshold": 1.0
        },
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight= -2.5) #0
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)

    # Height-based reward 
    height_scan_threshold = RewTerm(
        func=reward_base_height_above_threshold,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_height": 0.2,  # 0.0 reward at 23cm
            "max_height": 0.31   # 1.0 reward at 26cm
        },
    )

    foot_clearance = RewTerm(
        func=foot_clearance_reward,
        weight=0.2, 
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "target_height": 0.1,  # Target 10cm foot clearance
            "std": 0.05,             # Standard deviation for reward shaping
            "tanh_mult": 1.0        # Velocity sensitivity multiplier
        },
    )

    # Foot spread penalty - discourage feet from narrowing in past a minimum stance width
    foot_spread = RewTerm(
        func=foot_spread_reward,
        weight=-0.4,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_spread": 0.19,  # meters, FL/FR and RL/RR lateral distance
        },
    )

    # Gait reward
    gait_pattern = RewTerm(
        func=gait_reward,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "std": 0.1,
            "max_err": 0.2,
            "synced_feet_pair_names": [
                ["FL_foot", "RR_foot"],  # Diagonal pair 1: Front-Left with Rear-Right
                ["FR_foot", "RL_foot"]   # Diagonal pair 2: Front-Right with Rear-Left
            ]
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    log_avg_feet_air_time = CurrTerm(
        func=log_avg_feet_air_time,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )


##
# Environment configuration
##


@configclass
class Go2EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the go2 locomotion velocity-tracking environment."""

    # Scene settings
    scene: Go2SceneCfg = Go2SceneCfg(num_envs=32000, env_spacing=2.5, clone_in_fabric=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005  # 200fps
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physics = PhysxCfg(
            solver_type=1,  # TGS solver for better performance
            gpu_max_rigid_patch_count=5 * 2**17,
            gpu_max_rigid_contact_count=2**24,
            gpu_found_lost_pairs_capacity=2**22,
            gpu_found_lost_aggregate_pairs_capacity=2**26,
            gpu_total_aggregate_pairs_capacity=2**22,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
        )
        # update sensor update periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        # if self.scene.height_scanner is not None:
        #     self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class Go2EnvCfg_PLAY(Go2EnvCfg):
    """Configuration for playing/testing the trained go2 agent."""

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

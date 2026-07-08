# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
import isaaclab.utils.math as math_utils
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Custom reward functions
##

def reward_height_scan_above_threshold(env, sensor_cfg: SceneEntityCfg, min_height: float = 0.25, max_height: float = 0.34) -> torch.Tensor:
    """Simple linear reward for height scan values.

    Args:
        env: The environment instance.
        sensor_cfg: The sensor configuration.
        min_height: Minimum height for reward (default: 0.25m) -> reward = 0.0.
        max_height: Maximum height for reward (default: 0.34m) -> reward = 1.0.

    Returns:
        Reward tensor with values between 0.0 and 1.0.
    """
    # Get the sensor
    sensor = env.scene.sensors[sensor_cfg.name]

    # Calculate height scan: sensor_height - hit_point_z
    height_data = sensor.data.pos_w.torch[:, 2].unsqueeze(1) - sensor.data.ray_hits_w.torch[..., 2]

    # Calculate mean height for each environment
    mean_height = height_data.mean(dim=1)  # Shape: [num_envs]

    # Simple linear reward: 0.0 at min_height, 1.0 at max_height
    reward = torch.clamp((mean_height - min_height) / (max_height - min_height), 0.0, 1.0)

    return reward


def reward_base_height_above_threshold(env, asset_cfg: SceneEntityCfg, min_height: float = 0.25, max_height: float = 0.34) -> torch.Tensor:
    """Simple linear reward for the robot base height, without ray-casting.

    Reads the root height directly from the articulation's physics state instead of
    height-scan ray hits, so it only makes sense on flat/known-height ground (the
    scene here uses a flat plane at z=0). Much cheaper than the height-scan version
    since it avoids the per-step raycast against the terrain mesh.

    Args:
        env: The environment instance.
        asset_cfg: The robot asset configuration.
        min_height: Minimum height for reward (default: 0.25m) -> reward = 0.0.
        max_height: Maximum height for reward (default: 0.34m) -> reward = 1.0.

    Returns:
        Reward tensor with values between 0.0 and 1.0.
    """
    asset = env.scene[asset_cfg.name]

    base_height = asset.data.root_pos_w[:, 2]

    # Simple linear reward: 0.0 at min_height, 1.0 at max_height
    reward = torch.clamp((base_height - min_height) / (max_height - min_height), 0.0, 1.0)

    return reward


def foot_clearance_reward(env, asset_cfg: SceneEntityCfg, target_height: float = 0.1, std: float = 0.25, tanh_mult: float = 1.0) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground.

    Args:
        env: The environment instance.
        asset_cfg: The asset configuration for the robot.
        target_height: Target height for foot clearance in meters (default: 0.1m).
        std: Standard deviation for the exponential reward (default: 0.25).
        tanh_mult: Multiplier for the tanh function to modulate velocity (default: 1.0).

    Returns:
        Reward tensor encouraging foot clearance during swing phase.
    """
    # Get the robot asset
    asset = env.scene[asset_cfg.name]

    # Calculate foot height error (difference from target height)
    foot_z_target_error = torch.square(asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2] - target_height)

    # Use foot velocity to determine swing phase (moving feet get more reward). Uses the full
    # 3D velocity, not just horizontal XY, so pure vertical stepping-in-place (which has ~0
    # horizontal velocity) still counts as swinging instead of being scored as planted.
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :], dim=2))

    # Combine height error with velocity (only reward moving feet)
    reward = foot_z_target_error * foot_velocity_tanh

    # Return exponential reward (lower error = higher reward)
    return torch.exp(-torch.sum(reward, dim=1) / std)


def foot_spread_reward(env, asset_cfg: SceneEntityCfg, min_spread: float = 0.15) -> torch.Tensor:
    """Penalize the feet for standing narrower than a minimum lateral stance width.

    Measures the lateral distance between left/right foot pairs (FL/FR, RL/RR) in a
    yaw-only base frame and only penalizes when a pair is narrower than `min_spread`.
    This one-sided deadzone (unlike a symmetric hip joint_deviation_l1 penalty pulling
    toward the exact default pose) doesn't fight commanded lateral/strafing motion,
    which also needs the hips to swing away from their neutral position.

    Args:
        env: The environment instance.
        asset_cfg: The robot asset configuration.
        min_spread: Minimum lateral distance (in meters) between a foot pair before
            the penalty kicks in.

    Returns:
        Penalty tensor (>= 0), larger the narrower a foot pair gets below min_spread.
    """
    asset = env.scene[asset_cfg.name]

    # Resolve exact FL/FR/RL/RR foot body indices once and cache on the asset instance,
    # since find_bodies() does an expensive regex resolution over USD prim paths.
    foot_ids = getattr(asset, "_foot_spread_id_cache", None)
    if foot_ids is None:
        foot_ids, _ = asset.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
        asset._foot_spread_id_cache = foot_ids

    # Rotate foot positions into a yaw-only frame so spread is measured relative to the
    # robot's heading, not skewed by its current roll/pitch.
    heading_quat = math_utils.yaw_quat(asset.data.root_quat_w.torch).unsqueeze(1).expand(-1, 4, -1)
    rel_pos_w = asset.data.body_pos_w.torch[:, foot_ids, :] - asset.data.root_pos_w.torch.unsqueeze(1)
    foot_pos_b = math_utils.quat_apply_inverse(heading_quat, rel_pos_w)

    front_spread = torch.abs(foot_pos_b[:, 0, 1] - foot_pos_b[:, 1, 1])  # FL vs FR
    rear_spread = torch.abs(foot_pos_b[:, 2, 1] - foot_pos_b[:, 3, 1])  # RL vs RR

    front_penalty = torch.clamp(min_spread - front_spread, min=0.0)
    rear_penalty = torch.clamp(min_spread - rear_spread, min=0.0)

    return front_penalty + rear_penalty


def gait_reward(env, sensor_cfg: SceneEntityCfg, std: float = 0.25, max_err: float = 0.1,
                synced_feet_pair_names: list = None) -> torch.Tensor:
    """Gait enforcing reward for quadrupeds (trotting gait).

    This reward enforces proper gait patterns by:
    - Synchronizing diagonal pairs of feet (front-left with rear-right, front-right with rear-left)
    - Anti-synchronizing between the two diagonal pairs

    Args:
        env: The environment instance.
        sensor_cfg: The sensor configuration for contact detection.
        std: Standard deviation for reward shaping (default: 0.25).
        max_err: Maximum error clipping value (default: 0.1).
        synced_feet_pair_names: Pairs of feet that should be synchronized.

    Returns:
        Reward tensor encouraging proper trotting gait.
    """
    # Get sensor
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # Default diagonal pairs for trotting gait if not specified
    if synced_feet_pair_names is None:
        synced_feet_pair_names = [
            [".*front_left.*foot", ".*rear_right.*foot"],  # Diagonal pair 1
            [".*front_right.*foot", ".*rear_left.*foot"]   # Diagonal pair 2
        ]

    # Find body IDs for each foot pair. find_bodies() resolves USD prim paths via
    # regex, which is too expensive to redo every step, so cache the result on the
    # sensor instance (it's the same object across every call for the run's lifetime).
    synced_feet_pairs = getattr(contact_sensor, "_gait_pair_cache", None)
    if synced_feet_pairs is None:
        try:
            synced_feet_pair_0 = contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
            synced_feet_pair_1 = contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        except:
            # Fallback to simpler naming if the above doesn't work
            try:
                synced_feet_pair_0 = contact_sensor.find_bodies([".*FL.*foot", ".*RR.*foot"])[0]
                synced_feet_pair_1 = contact_sensor.find_bodies([".*FR.*foot", ".*RL.*foot"])[0]
            except:
                # Final fallback - just use all feet with basic indices
                all_feet = contact_sensor.find_bodies([".*foot"])[0]
                if len(all_feet) >= 4:
                    synced_feet_pair_0 = [all_feet[0], all_feet[3]]  # Front-left, rear-right
                    synced_feet_pair_1 = [all_feet[1], all_feet[2]]  # Front-right, rear-left
                else:
                    # Return zero reward if we can't find proper feet
                    return torch.zeros(env.num_envs, device=env.device)

        synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]
        contact_sensor._gait_pair_cache = synced_feet_pairs

    def sync_reward_func(foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = contact_sensor.data.current_air_time.torch
        contact_time = contact_sensor.data.current_contact_time.torch
        # Penalize difference between air and contact times of synced feet pairs
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=max_err**2)
        return torch.exp(-(se_air + se_contact) / std)

    def async_reward_func(foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = contact_sensor.data.current_air_time.torch
        contact_time = contact_sensor.data.current_contact_time.torch
        # Penalize when opposing feet are in same phase (should be opposite)
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / std)

    # Synchronous rewards (diagonal pairs should move together)
    sync_reward_0 = sync_reward_func(synced_feet_pairs[0][0], synced_feet_pairs[0][1])
    sync_reward_1 = sync_reward_func(synced_feet_pairs[1][0], synced_feet_pairs[1][1])
    sync_reward = sync_reward_0 * sync_reward_1

    # Asynchronous rewards (diagonal pairs should be opposite to each other)
    async_reward_0 = async_reward_func(synced_feet_pairs[0][0], synced_feet_pairs[1][0])
    async_reward_1 = async_reward_func(synced_feet_pairs[0][1], synced_feet_pairs[1][1])
    async_reward_2 = async_reward_func(synced_feet_pairs[0][0], synced_feet_pairs[1][1])
    async_reward_3 = async_reward_func(synced_feet_pairs[1][0], synced_feet_pairs[0][1])
    async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3

    # Always enforce gait synchronization, including when standing/stepping in place, so
    # diagonal-pair coordination is rewarded regardless of commanded speed.
    return sync_reward * async_reward


def feet_air_time_reward(env, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float,
                          cap: float | None = None, command_deadzone: float = 0.0) -> torch.Tensor:
    """Reward long-enough (but not too long) steps taken by the feet using an L2-kernel.

    Same as isaaclab_tasks' mdp.feet_air_time, except:
    - The hard >0.1 m/s command gate is replaced with an optional, much smaller deadzone
      (0.0 by default = no gate), so stepping in place at low/zero commanded velocity
      still earns reward instead of being zeroed out.
    - `cap`, if set, clamps the air time used in the reward so swings longer than `cap`
      earn no extra reward. Without a cap this reward is linear and unbounded in swing
      duration, meaning the policy is always better off taking *longer* strides no matter
      how low `threshold` is set. Capping removes that "longer is always better" pull and
      creates real pressure toward a fast, specific cadence instead.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    if cap is not None:
        last_air_time = torch.clamp(last_air_time, max=cap)
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    if command_deadzone > 0.0:
        reward *= torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_deadzone
    return reward

##
# Pre-defined configs
##
from .go2_articulation_cfg import MY_DOG_CFG  # Import your go2 articulation config


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
    # robot - using your custom go2 configuration
    robot: ArticulationCfg = MY_DOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Height scanner for terrain awareness
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,  # Enable ray visualization
        mesh_prim_paths=["/World/ground"],
    )

    # Contact sensors - commented out for now
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
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 2.5), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.35, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )

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
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
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
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
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
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.00002)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    feet_air_time = RewTerm(
        func=feet_air_time_reward,
        weight=0.3, #0.125
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.25, #0.5
            "cap": 0.35,  # no extra reward for swings longer than this -> pushes toward a fast, capped cadence
            "command_deadzone": 0.0,  # no gate: reward stepping even at zero commanded velocity
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"),
            "threshold": 1.0
        },
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight= -0.2) #0
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)

    # Height-based reward - simple linear reward system (uses root state, no height-scan raycast)
    height_scan_threshold = RewTerm(
        func=reward_base_height_above_threshold,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_height": 0.25,  # 0.0 reward at 23cm
            "max_height": 0.34   # 1.0 reward at 26cm
        },
    )

    # Foot clearance reward - encourage lifting feet during swing phase
    foot_clearance = RewTerm(
        func=foot_clearance_reward,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "target_height": 0.13,  # Target 10cm foot clearance
            "std": 0.05,             # Standard deviation for reward shaping
            "tanh_mult": 1.0        # Velocity sensitivity multiplier
        },
    )

    # Foot spread penalty - discourage feet from narrowing in past a minimum stance width
    foot_spread = RewTerm(
        func=foot_spread_reward,
        weight=-0.5,  # tune against logged foot spread; only fires when narrower than min_spread
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_spread": 0.2,  # meters, starting guess for FL/FR and RL/RR lateral distance
        },
    )

    # Gait reward - enforce proper trotting gait pattern
    gait_pattern = RewTerm(
        func=gait_reward,
        weight=1.0,  # Adjust this weight as needed
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "std": 0.1,             # Standard deviation for reward shaping (tighter control)
            "max_err": 0.2,         # Maximum error clipping
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
    # Contact-based termination commented out for now
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # No terrain curriculum for flat ground
    pass


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
        # PhysX settings optimized for your setup
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
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt


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

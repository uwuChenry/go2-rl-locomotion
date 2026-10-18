# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for the go2 locomotion environment."""

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils


def log_avg_feet_air_time(env, env_ids, sensor_cfg: SceneEntityCfg) -> float:
    """Reports mean feet air time for logging only -- does not affect reward.

    Wired in as a curriculum term (not a reward term) since RewardManager skips
    calling any term with weight=0.0, so a "silent" reward term can't be used
    for pure diagnostics. Curriculum terms run once per episode reset and their
    return value is logged to "Curriculum/<term_name>" without contributing to
    the reward signal.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.last_air_time.torch[env_ids][:, sensor_cfg.body_ids].mean().item()


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


# def foot_clearance_reward(env, asset_cfg: SceneEntityCfg, target_height: float = 0.1, min_velocity: float = 0.1) -> torch.Tensor:
#     """Reward feet that are simultaneously moving AND have cleared a minimum height.

#     Unlike a smoothly-shaped exp(-error/std) reward, this is a hard AND gate: a foot
#     only counts if it's actually swinging (not planted/dragging) AND its height is at
#     or above `target_height`. There's no partial credit for a foot that's fast but low
#     (e.g. dragging along the ground) or one that's high but not moving -- both earn zero.

#     Args:
#         env: The environment instance.
#         asset_cfg: The asset configuration for the robot.
#         target_height: Minimum foot height (in meters) to count as "cleared".
#         min_velocity: Minimum foot speed (in m/s, full 3D velocity) to count as
#             "moving" -- filters out stance-phase jitter so planted feet can't pass the gate.

#     Returns:
#         Reward tensor: count of feet simultaneously moving and cleared (0 to len(body_ids)).
#     """
#     # Get the robot asset
#     asset = env.scene[asset_cfg.name]

#     foot_height = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
#     # Full 3D velocity, not just horizontal XY, so pure vertical stepping-in-place still
#     # counts as "moving" instead of being scored as planted.
#     foot_speed = torch.norm(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :], dim=2)

#     is_moving = foot_speed > min_velocity
#     is_cleared = foot_height > target_height

#     return torch.sum((is_moving & is_cleared).float(), dim=1)

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

def foot_spread_reward(env, asset_cfg: SceneEntityCfg, min_spread: float = 0.2) -> torch.Tensor:
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


class FootStepQualityReward(ManagerTermBase):
    """Reward a foot once per landing if it was both well-timed AND cleared a minimum height.

    Replaces two separate terms (feet_air_time's timing reward and foot_clearance's
    per-step height gate) with a single event-triggered "good step" signal, paid out only
    at the instant a foot lands. This closes the double-dipping loophole where a policy
    could earn extra reward just by hovering a foot above the height bar for many steps
    (foot_clearance was summed every step) independent of how long it stayed airborne
    (feet_air_time plateaued past its cap but never penalized excessive hang time).

    Timing quality is a tent function of the completed swing's air time: 0 below
    `threshold`, ramping up to a peak of 1 at `cap`, then ramping back down to 0 at
    `hard_cap` (previously it just plateaued at `cap` with no ceiling). Height clearance
    is scored from the *peak* height reached during that specific swing (tracked in a
    persistent per-foot buffer, reset at liftoff and read out at landing) rather than the
    foot's height at any single instant, since height at the landing step itself is ~0.
    Clearance is a quadratic ramp (0 at zero height -> 1.0 at target_height, flat above),
    not a hard 0/1 cutoff -- easier to learn from early on since a foot that's getting
    close but hasn't fully cleared still earns partial credit instead of nothing.

    Args (passed as reward term params, forwarded to __call__ each step):
        sensor_cfg: The contact sensor configuration (used for air-time bookkeeping).
        asset_cfg: The robot asset configuration (used for foot world position).
        target_height: Minimum peak height (in meters) a swing must reach to count as cleared.
        threshold: Air time (in seconds) below which a landing earns zero timing reward.
        cap: Air time (in seconds) at which timing reward peaks at 1.0.
        hard_cap: Air time (in seconds) at which timing reward has decayed back to 0.0.
    """

    _FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    def __init__(self, cfg: RewTerm, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        # Resolve the same four feet, by exact name, independently on both the asset and
        # the sensor -- their body-id orderings aren't guaranteed to match each other, so
        # we can't reuse one SceneEntityCfg's body_ids to index into the other's tensors.
        self.asset_body_ids, _ = self.asset.find_bodies(self._FOOT_NAMES, preserve_order=True)
        self.sensor_body_ids, _ = self.contact_sensor.find_bodies(self._FOOT_NAMES, preserve_order=True)
        self.peak_height = torch.zeros(env.num_envs, len(self._FOOT_NAMES), device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.peak_height[env_ids] = 0.0

    def __call__(
        self,
        env,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        target_height: float = 0.1,
        threshold: float = 0.4,
        cap: float = 0.5,
        hard_cap: float = 0.65,
    ) -> torch.Tensor:
        foot_z = self.asset.data.body_pos_w.torch[:, self.asset_body_ids, 2]

        is_first_air = self.contact_sensor.compute_first_air(env.step_dt).torch[:, self.sensor_body_ids].bool()
        is_first_contact = self.contact_sensor.compute_first_contact(env.step_dt).torch[:, self.sensor_body_ids].bool()

        # Start a fresh peak at liftoff, then track the running max while airborne.
        self.peak_height = torch.where(is_first_air, torch.zeros_like(self.peak_height), self.peak_height)
        self.peak_height = torch.maximum(self.peak_height, foot_z)

        # Tent-shaped timing quality: ramps 0 -> 1 across [threshold, cap], then back down
        # to 0 across [cap, hard_cap], instead of plateauing at 1 forever past cap.
        last_air_time = self.contact_sensor.data.last_air_time.torch[:, self.sensor_body_ids]
        ramp_up = torch.clamp((last_air_time - threshold) / (cap - threshold), 0.0, 1.0)
        ramp_down = torch.clamp((hard_cap - last_air_time) / (hard_cap - cap), 0.0, 1.0)
        timing_quality = torch.minimum(ramp_up, ramp_down)

        # Quadratic clearance credit instead of a hard 0/1 gate: 0 at zero height, ramping
        # up to 1.0 as peak_height approaches target_height, flat at 1.0 once it's cleared.
        # Gives partial credit (and gradient signal) for a foot that's getting close but
        # hasn't fully cleared yet, instead of all-or-nothing -- easier to bootstrap early
        # in training than an immediate hard cutoff.
        clearance_ratio = torch.clamp(self.peak_height / target_height, 0.0, 1.0)
        clearance_score = clearance_ratio**2

        # Only pays out on the landing step for a foot that just completed a swing that
        # was both well-timed and had some/full clearance credit.
        return torch.sum(timing_quality * clearance_score * is_first_contact.float(), dim=1)


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
                          cap_start: float, cap_end: float, hard_cap: float,
                          command_deadzone: float = 0.0) -> torch.Tensor:
    """Reward well-timed steps taken by the feet using a trapezoidal quality score.

    Same as isaaclab_tasks' mdp.feet_air_time, except:
    - The hard >0.1 m/s command gate is replaced with an optional, much smaller deadzone
      (0.0 by default = no gate), so stepping in place at low/zero commanded velocity
      still earns reward instead of being zeroed out.
    - Instead of a linear, unbounded-above `(air_time - threshold)` (which always pulls
      the policy toward *longer* strides no matter how low `threshold` is set), air time
      is scored with a trapezoid: 0 below `threshold`, ramping up to 1 across
      [`threshold`, `cap_start`], flat at 1 across [`cap_start`, `cap_end`], then ramping
      back down to 0 across [`cap_end`, `hard_cap`]. Both too-short and too-long swings
      are actively pushed against instead of merely not being extra-rewarded.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    ramp_up = torch.clamp((last_air_time - threshold) / (cap_start - threshold), 0.0, 1.0)
    ramp_down = torch.clamp((hard_cap - last_air_time) / (hard_cap - cap_end), 0.0, 1.0)
    quality = torch.minimum(ramp_up, ramp_down)
    reward = torch.sum(torch.square(quality * first_contact), dim=1)
    if command_deadzone > 0.0:
        reward *= torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_deadzone
    return reward


def feet_air_time_uncapped(env, sensor_cfg: SceneEntityCfg, threshold: float, cap: float) -> torch.Tensor:
    """Same as isaaclab_tasks' mdp.feet_air_time, except:
    - No command gate: reward applies regardless of commanded velocity (matches
      feet_air_time_reward's command_deadzone=0.0 behavior).
    - Swings longer than `cap` earn zero instead of continuing to grow reward
      unboundedly past the threshold.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    reward = (last_air_time - threshold) * first_contact
    reward = torch.where(last_air_time > cap, torch.zeros_like(reward), reward)
    return torch.sum(reward, dim=1)


def feet_air_time_tent(env, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Symmetric, uncapped tent-shaped reward peaking at `threshold`.

    Unlike feet_air_time_uncapped (threshold = floor, reward grows unbounded above
    it), here `threshold` is the *target* swing duration: reward is 0 at exactly
    threshold and decreases linearly (unbounded, no cap) the further last_air_time
    lands from it in either direction -- both too-short and too-long swings are
    penalized at the same rate. No command gate, matching feet_air_time_uncapped.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    reward = -torch.abs(last_air_time - threshold) * first_contact
    return torch.sum(reward, dim=1)


def feet_air_time_tent_squared(env, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Squared version of :func:`feet_air_time_tent`.

    Identical in shape and sign -- 0 at exactly `threshold`, negative and unbounded on
    both sides -- but the penalty grows with the *square* of the error instead of its
    absolute value. The gradient is therefore proportional to the error rather than
    constant: near-target swings are barely penalized, while far-off swings are punished
    much harder (at 0.1 s of error the penalty is 10x smaller than the linear version,
    at 1.0 s it is 10x larger).

    Note the weight needs rescaling when switching over, since the error is squared: with
    threshold=0.2 the linear form pays about -0.2 for a dropped swing, the squared form
    about -0.04.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    reward = -torch.square(last_air_time - threshold) * first_contact
    return torch.sum(reward, dim=1)


class FootClearanceQualityReward(ManagerTermBase):
    """Reward a foot once per landing for the peak height it reached during its swing.

    Mirrors feet_air_time_reward's threshold -> cap ramp, but scores swing height instead
    of duration, and has no hard cap: reward ramps 0 -> 1 across [threshold, cap], then
    stays at 1.0 for any peak height above cap. A foot that clears extra high is never
    penalized for it, unlike feet_air_time_reward which does punish swings that run long.

    Peak height is tracked in a persistent per-foot buffer, reset at liftoff and updated
    with a running max while airborne, then read out and paid at the landing step --
    scoring the foot's *current* height wouldn't work since it's ~0 right at landing.

    Args (passed as reward term params, forwarded to __call__ each step):
        asset_cfg: The robot asset configuration (used for foot world position).
        sensor_cfg: The contact sensor configuration (used for landing detection).
        threshold: Peak swing height (in meters) below which a landing earns zero reward.
        cap: Peak swing height (in meters) at which the reward saturates at 1.0.
    """

    _FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    def __init__(self, cfg: RewTerm, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        # Resolve the same four feet, by exact name, independently on both the asset and
        # the sensor -- their body-id orderings aren't guaranteed to match each other.
        self.asset_body_ids, _ = self.asset.find_bodies(self._FOOT_NAMES, preserve_order=True)
        self.sensor_body_ids, _ = self.contact_sensor.find_bodies(self._FOOT_NAMES, preserve_order=True)
        self.peak_height = torch.zeros(env.num_envs, len(self._FOOT_NAMES), device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.peak_height[env_ids] = 0.0

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        threshold: float = 0.03,
        cap: float = 0.1,
    ) -> torch.Tensor:
        foot_z = self.asset.data.body_pos_w.torch[:, self.asset_body_ids, 2]

        is_first_air = self.contact_sensor.compute_first_air(env.step_dt).torch[:, self.sensor_body_ids].bool()
        is_first_contact = self.contact_sensor.compute_first_contact(env.step_dt).torch[:, self.sensor_body_ids].bool()

        # Start a fresh peak at liftoff, then track the running max while airborne.
        self.peak_height = torch.where(is_first_air, torch.zeros_like(self.peak_height), self.peak_height)
        self.peak_height = torch.maximum(self.peak_height, foot_z)

        quality = torch.clamp((self.peak_height - threshold) / (cap - threshold), 0.0, 1.0)

        return torch.sum(quality * is_first_contact.float(), dim=1)

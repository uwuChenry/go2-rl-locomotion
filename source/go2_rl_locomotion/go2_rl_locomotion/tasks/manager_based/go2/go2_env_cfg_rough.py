# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rough-terrain go2 locomotion with a height-scan ("scan dots") observation.

Builds on :class:`Go2EnvCfg` and follows Isaac Lab's
``isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`` rough setup:
generated terrain with a velocity-based terrain-level curriculum, plus a yaw-aligned grid
ray caster whose heights are fed to the policy.
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .go2_env_cfg import CurriculumCfg, Go2EnvCfg, Go2SceneCfg, ObservationsCfg

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class Go2RoughSceneCfg(Go2SceneCfg):
    """Go2 scene on generated rough terrain with a height scanner."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
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

    # scan dots: 17 x 11 = 187 rays on a 1.6m x 1.0m grid, centered on the base and
    # rotated with its yaw only, cast straight down from 20m above
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


##
# MDP settings
##


@configclass
class RoughObservationsCfg(ObservationsCfg):
    """Flat-env observations plus the height scan."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """Observations for policy group."""

        # base height minus terrain height at each scan dot, minus a 0.5m offset
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RoughCurriculumCfg(CurriculumCfg):
    """Flat-env curriculum plus terrain-level promotion/demotion."""

    # promote after walking > half the tile, demote after covering < half the commanded distance
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class Go2RoughEnvCfg(Go2EnvCfg):
    """Configuration for go2 velocity tracking on rough terrain with scan dots."""

    scene: Go2RoughSceneCfg = Go2RoughSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: RoughObservationsCfg = RoughObservationsCfg()
    curriculum: RoughCurriculumCfg = RoughCurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # post init of parent
        super().__post_init__()

        # scale down the terrains because the robot is small (same as Isaac Lab's go2 rough cfg)
        sub_terrains = self.scene.terrain.terrain_generator.sub_terrains
        sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        sub_terrains["random_rough"].noise_step = 0.01

        # generate terrains with increasing difficulty per row when the curriculum is on
        self.scene.terrain.terrain_generator.curriculum = getattr(self.curriculum, "terrain_levels", None) is not None

        # the scan dots are consumed once per policy step
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # -- rewards
        # Isaac Lab's default air-time reward instead of the flat env's tent shaping
        self.rewards.feet_air_time2 = None
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time,
            weight=0.125,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "command_name": "base_velocity",
                "threshold": 0.5,
            },
        )
        # these read world z and assume a ground plane at z=0, which no longer holds
        self.rewards.height_scan_threshold = None
        self.rewards.foot_clearance = None
        self.rewards.foot_clearance_quality = None


@configclass
class Go2RoughEnvCfg_PLAY(Go2RoughEnvCfg):
    """Configuration for playing/testing the trained go2 rough-terrain agent."""

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        # visualize the scan dots
        self.scene.height_scanner.debug_vis = True

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

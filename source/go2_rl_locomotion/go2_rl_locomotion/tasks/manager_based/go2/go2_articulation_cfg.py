# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg, PhysxRigidBodyPropertiesCfg

##
# Configuration for the Unitree Go2 (simplified) robot
##

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
GO2_SIMPLIFIED_USD_PATH = os.path.join(_ASSETS_DIR, "go2_simplified", "go2_simplified.usd")
GO2_BACK_PLATFORM_USD_PATH = os.path.join(
    _ASSETS_DIR, "go2_back_platform", "go2_back_platform.usd", "go2_back_platform", "go2_back_platform.usda"
)
GO2_SIMPLIFIED_BACK_PLATFORM_USD_PATH = os.path.join(
    _ASSETS_DIR,
    "go2_simplified_back_platform",
    "go2_simplified_back_platform",
    "go2_simplified_back_platform.usda",
)
GO2_COMBINED_USD_PATH = os.path.join(_ASSETS_DIR, "go2_combined", "go2_combined.usda")

MY_DOG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GO2_SIMPLIFIED_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=PhysxRigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=PhysxArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.00,
            ".*R_hip_joint": -0.00,
            "F[L,R]_thigh_joint": 0.72,
            "R[L,R]_thigh_joint": 0.72,
            ".*_calf_joint": -1.44,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)

##
# Configuration for the Unitree Go2 with the back platform payload
# (Jetson + rail plate + RPLidar + OAK camera, merged rigidly into the base body)
##

GO2_BACK_PLATFORM_CFG = MY_DOG_CFG.replace(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GO2_BACK_PLATFORM_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=PhysxRigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=PhysxArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
)

##
# Configuration for the Unitree Go2 (simplified) with the back platform payload
# (Jetson + rail plate + RPLidar + LiDAR + OAK camera) attached to the go2_simplified base link
#
# NOTE: this one comes from the go2_simplified_back_platform URDF import, which PhysX
# rejects (the converter emitted unnormalized link orientations). Use GO2_COMBINED_CFG.
##

GO2_SIMPLIFIED_BACK_PLATFORM_CFG = MY_DOG_CFG.replace(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GO2_SIMPLIFIED_BACK_PLATFORM_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=PhysxRigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=PhysxArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
)

##
# Configuration for the Unitree Go2 (simplified) with the back platform payload, built on
# top of the known-good go2_simplified.usd instead of a URDF import.
#
# The payload (jetson plate + rail plate + RPLidar + Livox lidar + OAK camera, 3.74 kg) is
# rigidly merged into the base link: visual-only geometry plus the combined mass, COM and
# inertia. The articulation itself is untouched, so the body/joint names are identical to
# MY_DOG_CFG and a policy trained on one transfers to the other. See
# scripts/build_go2_combined.py to regenerate the asset.
##

GO2_COMBINED_CFG = MY_DOG_CFG.replace(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GO2_COMBINED_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=PhysxRigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=PhysxArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
)

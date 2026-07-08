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

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Privileged observation terms (simulator ground truth the real robot cannot measure)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class foot_friction(ManagerTermBase):
    """Static friction coefficient of each selected body's first collision shape, one column per body.

    With the ground at friction 1.0 and ``friction_combine_mode="multiply"`` (see ``Go2SceneCfg``) this is exactly
    the effective foot-ground friction, which is what WTW's estimator predicts.

    PhysX keeps materials on the CPU, so the values are read once and cached on the env device. They are read on
    the first environment reset, because this term is built (and reset once) *before* the startup events run, so
    anything earlier would capture the pre-randomization materials. This assumes friction is only
    randomized at startup, which is what ``randomize_rigid_body_material`` recommends; re-reading every reset
    would cost a full CPU material copy per step.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]

        # first shape index of every selected body; same parsing as randomize_rigid_body_material
        num_shapes_per_body = [
            self.asset._physics_sim_view.create_rigid_body_view(link_path).max_shapes
            for link_path in self.asset.root_view.link_paths[0]
        ]
        body_ids = range(self.asset.num_bodies) if asset_cfg.body_ids == slice(None) else asset_cfg.body_ids
        self.shape_ids = []
        for body_id in body_ids:
            if num_shapes_per_body[body_id] == 0:
                raise ValueError(f"Body '{self.asset.body_names[body_id]}' has no collision shapes to read friction from.")
            self.shape_ids.append(sum(num_shapes_per_body[:body_id]))

        self.friction = torch.zeros(env.num_envs, len(self.shape_ids), device=env.device)
        self._loaded = False

    def reset(self, env_ids=None) -> None:
        # the ObservationManager also calls reset() from its own constructor, i.e. before the startup events have
        # randomized anything; skip reads until the manager exists
        if not self._loaded and hasattr(self._env, "observation_manager"):
            materials = wp.to_torch(self.asset.root_view.get_material_properties())
            self.friction[:] = materials[:, self.shape_ids, 0].to(self.friction.device)
            self._loaded = True

    def __call__(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        return self.friction

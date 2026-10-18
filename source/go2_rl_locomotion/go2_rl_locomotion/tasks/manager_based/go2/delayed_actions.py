# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint position action with a configurable actuation delay.

Isaac Lab applies actions instantly; a real robot (and a ROS2/Gazebo stack) does not. Walk
These Ways models a constant ~20 ms delay on the Go1, which at a 50 Hz control rate is
exactly one control step. Unmodelled latency is one of the larger sim-to-real gaps for
locomotion and no amount of observation noise substitutes for it.

This module imports pxr transitively, so it must not be imported at config-resolution
time -- see :mod:`delayed_actions_cfg` for why and how that is avoided.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .delayed_actions_cfg import DelayedJointPositionActionCfg


class DelayedJointPositionAction(JointPositionAction):
    """Joint position action that applies the action from ``delay`` control steps ago.

    The delay is sampled per-environment on reset from
    :attr:`~.delayed_actions_cfg.DelayedJointPositionActionCfg.delay_steps`, so a range
    models a robot whose latency varies run to run, while a degenerate range like
    ``(1, 1)`` pins it to a fixed one-step (20 ms at 50 Hz) delay.

    Only the *actuation* is delayed. The ``last_action`` observation still reports what the
    policy commanded this step, which is correct -- the policy knows its own output, it just
    does not see the effect yet.
    """

    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._min_delay, self._max_delay = int(cfg.delay_steps[0]), int(cfg.delay_steps[1])
        if self._min_delay < 0 or self._max_delay < self._min_delay:
            raise ValueError(f"Invalid delay_steps {cfg.delay_steps}: need 0 <= min <= max.")

        # ring buffer of the last (max_delay + 1) processed actions, newest at _head
        self._buffer = torch.zeros(self._max_delay + 1, self.num_envs, self.action_dim, device=self.device)
        self._head = 0
        self._delay = torch.randint(
            self._min_delay, self._max_delay + 1, (self.num_envs,), device=self.device, dtype=torch.long
        )
        self._env_indices = torch.arange(self.num_envs, device=self.device)
        self._delayed_actions = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor):
        # scale/offset/clip exactly as the base class does, then push onto the ring buffer
        super().process_actions(actions)
        self._head = (self._head + 1) % self._buffer.shape[0]
        self._buffer[self._head] = self._processed_actions
        # read back the entry `delay` steps behind the head, per environment
        read_idx = (self._head - self._delay) % self._buffer.shape[0]
        self._delayed_actions = self._buffer[read_idx, self._env_indices]

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._delayed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
            num_reset = self.num_envs
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            num_reset = env_ids.numel()
        # prime the buffer with the neutral (default-pose) action so a fresh episode never
        # actuates on a stale command left over from the previous one
        neutral = self._offset[env_ids] if isinstance(self._offset, torch.Tensor) else self._offset
        self._buffer[:, env_ids] = neutral
        self._delayed_actions[env_ids] = neutral
        # resample the per-environment latency
        if self._max_delay > self._min_delay:
            self._delay[env_ids] = torch.randint(
                self._min_delay, self._max_delay + 1, (num_reset,), device=self.device, dtype=torch.long
            )

    @property
    def delayed_actions(self) -> torch.Tensor:
        """The actions actually sent to the articulation this step."""
        return self._delayed_actions

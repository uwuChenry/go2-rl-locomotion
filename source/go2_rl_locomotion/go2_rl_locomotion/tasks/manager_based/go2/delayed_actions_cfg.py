# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the delayed joint position action.

Kept separate from :mod:`delayed_actions` on purpose. The runtime action class imports
``isaaclab.envs.mdp.actions.joint_actions``, which pulls in ``pxr`` -- and the training
scripts resolve the env config *before* ``SimulationApp`` starts. Importing pxr that early
crashes Kit with ``free(): invalid pointer``.

``class_type`` is therefore a lazy ``"{DIR}.module:Class"`` string. :func:`configclass`
wraps it in a ``ResolvableString`` that only imports the module when the ActionManager
actually instantiates the term, which is safely after the app has launched.
"""

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils.configclass import configclass


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`~.delayed_actions.DelayedJointPositionAction`."""

    class_type: str = "{DIR}.delayed_actions:DelayedJointPositionAction"

    delay_steps: tuple[int, int] = (1, 1)
    """Inclusive range of control steps to delay actuation by, sampled per environment on
    reset. Defaults to a fixed 1 step, which is WTW's ~20 ms at a 50 Hz control rate."""

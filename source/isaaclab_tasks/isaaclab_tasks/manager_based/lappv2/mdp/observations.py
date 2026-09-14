# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def _feet_contact(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
    return (torch.linalg.vector_norm(force, dim=-1) > 1.0).float()


def feet_contact_state(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return binary contact state for the four feet."""
    return _feet_contact(env, sensor_cfg)


def backflip_angle(env) -> torch.Tensor:
    """Return accumulated body-Y rotation toward the negative backflip direction."""
    if not hasattr(env, "_backflip_angle"):
        env._backflip_angle = torch.zeros(env.num_envs, device=env.device)
    return env._backflip_angle.unsqueeze(-1)

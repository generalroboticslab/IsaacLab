# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


def reset_backflip_state(env, env_ids: torch.Tensor) -> None:
    """Reset the fixed backflip state buffers for selected environments."""
    if not hasattr(env, "_backflip_angle"):
        env._backflip_angle = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_backflip_was_airborne"):
        env._backflip_was_airborne = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_backflip_success"):
        env._backflip_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_backflip_landing_event"):
        env._backflip_landing_event = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if isinstance(env_ids, slice):
        env._backflip_angle[env_ids] = 0.0
        env._backflip_was_airborne[env_ids] = False
        env._backflip_success[env_ids] = False
        env._backflip_landing_event[env_ids] = False
    else:
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        env._backflip_angle[ids] = 0.0
        env._backflip_was_airborne[ids] = False
        env._backflip_success[ids] = False
        env._backflip_landing_event[ids] = False

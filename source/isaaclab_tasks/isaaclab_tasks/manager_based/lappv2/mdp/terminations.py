# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


def backflip_failure(
    env,
    minimum_height: float = 0.18,
    maximum_tilt: float = 0.85,
    maximum_displacement: float = 2.5,
) -> torch.Tensor:
    """Terminate fallen or excessively displaced environments."""
    robot = env.scene["robot"]
    return (
        (robot.data.root_pos_w.torch[:, 2] < minimum_height)
        | (torch.linalg.vector_norm(robot.data.projected_gravity_b.torch[:, :2], dim=1) > maximum_tilt)
        | (torch.linalg.vector_norm(robot.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2], dim=1) > maximum_displacement)
    )

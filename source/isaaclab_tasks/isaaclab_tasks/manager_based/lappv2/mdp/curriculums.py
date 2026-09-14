# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

REWARD_STAGES = (
    (0.10, 1.75, 2.25, 4.00, 2.50, 0.50, 0.05, 0.00, 0.10, 0.50, 0.50, 0.75, -0.20, -0.002, -0.010, 0.50, -5.0),
    (0.10, 1.00, 2.75, 4.50, 3.00, 1.25, 0.25, 0.75, 0.30, 0.75, 1.00, 0.75, -0.25, -0.002, -0.010, 0.60, -5.0),
    (0.10, 0.50, 2.00, 3.50, 2.00, 1.75, 2.50, 1.75, 1.50, 1.00, 2.00, 1.00, -0.30, -0.0025, -0.012, 0.75, -5.0),
    (0.05, 0.25, 1.25, 2.50, 1.25, 1.00, 3.50, 1.25, 1.00, 4.00, 8.00, 2.50, -0.60, -0.003, -0.015, 1.00, -7.5),
    (0.05, 0.15, 1.00, 2.50, 1.25, 0.75, 3.00, 1.00, 0.75, 5.00, 10.00, 3.00, -0.75, -0.003, -0.015, 1.00, -10.0),
)

REWARD_NAMES = (
    "alive", "crouch", "launch", "takeoff", "height", "airborne", "rotation", "spin", "tuck",
    "landing", "success", "upright", "horizontal", "energy", "smooth", "symmetry", "failure",
)

# Stage 1 is the initial RewardsCfg row. These are the step thresholds for stages 2-5.
STAGE_STEPS = (100_000, 200_000, 300_000, 400_000)

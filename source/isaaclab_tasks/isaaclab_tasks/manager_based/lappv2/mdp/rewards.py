# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch

from isaaclab.managers import SceneEntityCfg


def _robot(env):
    return env.scene["robot"]


def _contacts(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
    return torch.linalg.vector_norm(force, dim=-1) > 1.0


def _ensure_state(env) -> None:
    if not hasattr(env, "_backflip_angle"):
        env._backflip_angle = torch.zeros(env.num_envs, device=env.device)
        env._backflip_was_airborne = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._backflip_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _pose_target(env, thigh: float, calf: float) -> torch.Tensor:
    target = _robot(env).data.default_joint_pos.torch[0].expand(env.num_envs, -1).clone()
    for joint_id, name in enumerate(_robot(env).joint_names):
        if "hip" in name:
            target[:, joint_id] = 0.0
        elif "thigh" in name:
            target[:, joint_id] = thigh
        elif "calf" in name:
            target[:, joint_id] = calf
    return target


def alive(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def crouch(env) -> torch.Tensor:
    error = _robot(env).data.joint_pos.torch - _pose_target(env, 1.45, -2.45)
    return torch.exp(-torch.mean(error.square(), dim=1) / 0.35)


def launch(env) -> torch.Tensor:
    error = _robot(env).data.joint_pos.torch - _pose_target(env, 0.25, -0.85)
    return torch.exp(-torch.mean(error.square(), dim=1) / 0.35)


def takeoff(env) -> torch.Tensor:
    return torch.sigmoid((_robot(env).data.root_lin_vel_w.torch[:, 2] - 0.5) / 0.5)


def height(env, desired_height: float = 0.85) -> torch.Tensor:
    return torch.exp(-torch.abs(_robot(env).data.root_pos_w.torch[:, 2] - desired_height) / 0.35)


def airborne(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    return (~torch.any(_contacts(env, sensor_cfg), dim=1)).float()


def rotation(env) -> torch.Tensor:
    _ensure_state(env)
    env._backflip_angle += _robot(env).data.root_ang_vel_b.torch[:, 1] * env.step_dt
    return torch.exp(-torch.abs(env._backflip_angle + 2.0 * math.pi) / math.pi)


def spin(env, target_spin_rate: float = -8.0) -> torch.Tensor:
    return torch.exp(-torch.abs(_robot(env).data.root_ang_vel_b.torch[:, 1] - target_spin_rate) / 4.0)


def tuck(env) -> torch.Tensor:
    error = _robot(env).data.joint_pos.torch - _pose_target(env, 1.45, -2.45)
    return torch.exp(-torch.mean(error.square(), dim=1) / 0.75)


def _landing_event(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    _ensure_state(env)
    contact = _contacts(env, sensor_cfg)
    grounded = torch.any(contact, dim=1)
    upright = torch.linalg.vector_norm(_robot(env).data.projected_gravity_b.torch[:, :2], dim=1) < 0.7
    low_horizontal_speed = torch.linalg.vector_norm(_robot(env).data.root_lin_vel_w.torch[:, :2], dim=1) < 1.0
    event = env._backflip_was_airborne & grounded & (torch.abs(env._backflip_angle + 2.0 * math.pi) < 0.45)
    event &= upright & low_horizontal_speed
    env._backflip_was_airborne = ~grounded
    env._backflip_landing_event = event
    return event.float()


def landing(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    return _landing_event(env, sensor_cfg)


def success(env) -> torch.Tensor:
    _ensure_state(env)
    event = getattr(env, "_backflip_landing_event", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    value = event & ~env._backflip_success
    env._backflip_success |= value
    return value.float()


def upright(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    grounded = torch.any(_contacts(env, sensor_cfg), dim=1)
    quality = torch.exp(-torch.sum(_robot(env).data.projected_gravity_b.torch[:, :2].square(), dim=1) / 0.15)
    return quality * grounded.float()


def horizontal(env) -> torch.Tensor:
    return torch.sum(_robot(env).data.root_lin_vel_w.torch[:, :2].square(), dim=1)


def energy(env) -> torch.Tensor:
    return torch.mean(env.action_manager.action.square(), dim=1)


def smooth(env) -> torch.Tensor:
    return torch.mean((env.action_manager.action - env.action_manager.prev_action).square(), dim=1)


def symmetry(env) -> torch.Tensor:
    joint_pos = _robot(env).data.joint_pos.torch
    error = torch.zeros(env.num_envs, device=env.device)
    names = _robot(env).joint_names
    for part in ("hip", "thigh", "calf"):
        left = [i for i, name in enumerate(names) if "L" in name and part in name]
        right = [i for i, name in enumerate(names) if "R" in name and part in name]
        if left and right:
            error += torch.abs(joint_pos[:, left[0]] - joint_pos[:, right[0]])
    return torch.exp(-error / 0.5)


def failure(env, minimum_height: float = 0.18, maximum_tilt: float = 0.85, maximum_displacement: float = 2.5) -> torch.Tensor:
    robot = _robot(env)
    return (
        (robot.data.root_pos_w.torch[:, 2] < minimum_height)
        | (torch.linalg.vector_norm(robot.data.projected_gravity_b.torch[:, :2], dim=1) > maximum_tilt)
        | (torch.linalg.vector_norm(robot.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2], dim=1) > maximum_displacement)
    ).float()

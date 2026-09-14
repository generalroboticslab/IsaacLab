# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from . import mdp
from .mdp.curriculums import REWARD_NAMES, REWARD_STAGES, STAGE_STEPS


def _reward_weight_stage(stage: int, reward_index: int) -> CurrTerm:
    """Create one standard step-wise reward-weight curriculum term."""
    return CurrTerm(
        func=base_mdp.modify_reward_weight,
        params={
            "term_name": REWARD_NAMES[reward_index],
            "weight": REWARD_STAGES[stage][reward_index],
            "num_steps": STAGE_STEPS[stage - 1],
        },
    )


@configclass
class Lappv2SceneCfg(InteractiveSceneCfg):
    """Fixed Go2 scene shared by every curriculum stage."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)))
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    feet = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8)),
    )


@configclass
class ActionsCfg:
    """Fixed 12-joint position action interface."""

    joint_position = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Fixed policy observation interface."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
        actions = ObsTerm(func=base_mdp.last_action)
        angle = ObsTerm(func=mdp.backflip_angle)
        feet_contact = ObsTerm(
            func=mdp.feet_contact_state,
            params={"sensor_cfg": SceneEntityCfg("feet", body_names=".*_foot")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Fixed reset interface."""

    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )
    reset_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_backflip = EventTerm(func=mdp.reset_backflip_state, mode="reset")


@configclass
class RewardsCfg:
    """The fixed 17-term reward basis; the curriculum changes only these weights."""

    alive = RewTerm(func=mdp.alive, weight=0.10)
    crouch = RewTerm(func=mdp.crouch, weight=1.75)
    launch = RewTerm(func=mdp.launch, weight=2.25)
    takeoff = RewTerm(func=mdp.takeoff, weight=4.00)
    height = RewTerm(func=mdp.height, weight=2.50)
    airborne = RewTerm(func=mdp.airborne, weight=0.50, params={"sensor_cfg": SceneEntityCfg("feet", body_names=".*_foot")})
    rotation = RewTerm(func=mdp.rotation, weight=0.05)
    spin = RewTerm(func=mdp.spin, weight=0.00)
    tuck = RewTerm(func=mdp.tuck, weight=0.10)
    landing = RewTerm(func=mdp.landing, weight=0.50, params={"sensor_cfg": SceneEntityCfg("feet", body_names=".*_foot")})
    success = RewTerm(func=mdp.success, weight=0.50)
    upright = RewTerm(func=mdp.upright, weight=0.75, params={"sensor_cfg": SceneEntityCfg("feet", body_names=".*_foot")})
    horizontal = RewTerm(func=mdp.horizontal, weight=-0.20)
    energy = RewTerm(func=mdp.energy, weight=-0.002)
    smooth = RewTerm(func=mdp.smooth, weight=-0.010)
    symmetry = RewTerm(func=mdp.symmetry, weight=0.50)
    failure = RewTerm(func=mdp.failure, weight=-5.0)


@configclass
class TerminationsCfg:
    """Fixed failure and timeout semantics."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    failure = DoneTerm(
        func=mdp.backflip_failure,
        params={"minimum_height": 0.18, "maximum_tilt": 0.85, "maximum_displacement": 2.5},
    )


@configclass
class CurriculumCfg:
    """Step-wise reward-weight schedules for stages 2 through 5."""

    stage2_alive = _reward_weight_stage(1, 0)
    stage2_crouch = _reward_weight_stage(1, 1)
    stage2_launch = _reward_weight_stage(1, 2)
    stage2_takeoff = _reward_weight_stage(1, 3)
    stage2_height = _reward_weight_stage(1, 4)
    stage2_airborne = _reward_weight_stage(1, 5)
    stage2_rotation = _reward_weight_stage(1, 6)
    stage2_spin = _reward_weight_stage(1, 7)
    stage2_tuck = _reward_weight_stage(1, 8)
    stage2_landing = _reward_weight_stage(1, 9)
    stage2_success = _reward_weight_stage(1, 10)
    stage2_upright = _reward_weight_stage(1, 11)
    stage2_horizontal = _reward_weight_stage(1, 12)
    stage2_energy = _reward_weight_stage(1, 13)
    stage2_smooth = _reward_weight_stage(1, 14)
    stage2_symmetry = _reward_weight_stage(1, 15)
    stage2_failure = _reward_weight_stage(1, 16)

    stage3_alive = _reward_weight_stage(2, 0)
    stage3_crouch = _reward_weight_stage(2, 1)
    stage3_launch = _reward_weight_stage(2, 2)
    stage3_takeoff = _reward_weight_stage(2, 3)
    stage3_height = _reward_weight_stage(2, 4)
    stage3_airborne = _reward_weight_stage(2, 5)
    stage3_rotation = _reward_weight_stage(2, 6)
    stage3_spin = _reward_weight_stage(2, 7)
    stage3_tuck = _reward_weight_stage(2, 8)
    stage3_landing = _reward_weight_stage(2, 9)
    stage3_success = _reward_weight_stage(2, 10)
    stage3_upright = _reward_weight_stage(2, 11)
    stage3_horizontal = _reward_weight_stage(2, 12)
    stage3_energy = _reward_weight_stage(2, 13)
    stage3_smooth = _reward_weight_stage(2, 14)
    stage3_symmetry = _reward_weight_stage(2, 15)
    stage3_failure = _reward_weight_stage(2, 16)

    stage4_alive = _reward_weight_stage(3, 0)
    stage4_crouch = _reward_weight_stage(3, 1)
    stage4_launch = _reward_weight_stage(3, 2)
    stage4_takeoff = _reward_weight_stage(3, 3)
    stage4_height = _reward_weight_stage(3, 4)
    stage4_airborne = _reward_weight_stage(3, 5)
    stage4_rotation = _reward_weight_stage(3, 6)
    stage4_spin = _reward_weight_stage(3, 7)
    stage4_tuck = _reward_weight_stage(3, 8)
    stage4_landing = _reward_weight_stage(3, 9)
    stage4_success = _reward_weight_stage(3, 10)
    stage4_upright = _reward_weight_stage(3, 11)
    stage4_horizontal = _reward_weight_stage(3, 12)
    stage4_energy = _reward_weight_stage(3, 13)
    stage4_smooth = _reward_weight_stage(3, 14)
    stage4_symmetry = _reward_weight_stage(3, 15)
    stage4_failure = _reward_weight_stage(3, 16)

    stage5_alive = _reward_weight_stage(4, 0)
    stage5_crouch = _reward_weight_stage(4, 1)
    stage5_launch = _reward_weight_stage(4, 2)
    stage5_takeoff = _reward_weight_stage(4, 3)
    stage5_height = _reward_weight_stage(4, 4)
    stage5_airborne = _reward_weight_stage(4, 5)
    stage5_rotation = _reward_weight_stage(4, 6)
    stage5_spin = _reward_weight_stage(4, 7)
    stage5_tuck = _reward_weight_stage(4, 8)
    stage5_landing = _reward_weight_stage(4, 9)
    stage5_success = _reward_weight_stage(4, 10)
    stage5_upright = _reward_weight_stage(4, 11)
    stage5_horizontal = _reward_weight_stage(4, 12)
    stage5_energy = _reward_weight_stage(4, 13)
    stage5_smooth = _reward_weight_stage(4, 14)
    stage5_symmetry = _reward_weight_stage(4, 15)
    stage5_failure = _reward_weight_stage(4, 16)


@configclass
class Lappv2EnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based Go2 backflip environment."""

    scene: Lappv2SceneCfg = Lappv2SceneCfg(num_envs=2048, env_spacing=3.0, clone_in_fabric=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 5.0
        self.sim.dt = 1 / 200.0
        self.sim.render_interval = self.decimation

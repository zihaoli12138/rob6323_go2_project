# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0

    # spaces
    action_scale = 0.25
    action_space = 12
    observation_space = 48 + 4  # baseline adds 4 clock inputs
    state_space = 0
    debug_vis = True

    # ---------------------------
    # Rewards baseline
    # ---------------------------
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5

    # baseline: action smoothness (note: negative; TA suggests -0.0001 or smaller)
    action_rate_reward_scale = -0.0001

    # baseline: Raibert heuristic (note: negative; set to 0.0 to disable)
    raibert_heuristic_reward_scale = -20.0

    # your shaping (set to 0.0 to disable)
    base_level_reward_scale = 0.10
    base_height_reward_scale = 0.60

    # exp mapping denominators (bigger => gentler)
    base_level_exp_denom = 0.25
    base_height_exp_denom = 0.02

    # base height target for shaping reward (world z)
    base_height_target = 0.35

    # termination threshold
    base_height_min = 0.22  # terminate if base < 22 cm

    # ---------------------------
    # Anti-hop regularizers
    # ---------------------------
    lin_vel_z_reward_scale = -2.0      # (v_z)^2
    ang_vel_xy_reward_scale = -0.05    # (ωx^2 + ωy^2)
    torque_reward_scale = -2.0e-5      # sum(τ^2)  (VERY small)
    dof_vel_reward_scale = -1.0e-4     # sum(qd^2) (small)

    # ---------------------------
    # TA rewards: feet clearance + swing-contact force
    # ---------------------------

    # Feet clearance: match a swing-foot height profile (penalty, so negative)
    feet_clearance_reward_scale = -0.05

    # Penalize foot contact force during swing (penalty, so negative)
    tracking_contacts_shaped_force_reward_scale = -0.02

    # Swing foot height profile (meters)
    foot_clearance_peak = 0.08   # peak extra height during swing
    foot_radius_offset = 0.02    # offset for foot radius (~2cm)

    # Contact schedule smoothing (used to compute desired_contact_states smoothly)
    contact_smoothing_kappa = 0.07

    # Denominator in exp(-force^2 / denom) for shaped-force penalty
    contact_force_exp_denom = 100.0

    # ---------------------------
    # Command sampling
    # ---------------------------
    command_lin_vel_x_range = (-1.0, 1.0)
    command_lin_vel_y_range = (-0.05, 0.05)
    command_yaw_rate_range  = (-1.0, 1.0)

    # ---------------------------
    # Simulation
    # ---------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # ---------------------------
    # Robot: disable implicit PD; we do torque-level PD
    # ---------------------------
    Kp = 20.0
    Kd = 0.5
    torque_limits = 100.0

    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,  # disable implicit P-gain
        damping=0.0,    # disable implicit D-gain
    )

    # ---------------------------
    # Scene / sensors / debug markers
    # ---------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )

    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

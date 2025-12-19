# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

@configclass
class EventCfg:
    """Configuration for randomization."""
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 20.0
    action_scale = 0.25
    action_space = 12
    # Base (48) + Clock (4) = 52
    observation_space = 52 
    debug_vis = True

    # --- TA Specified Reward Scales ---
    feet_clearance_reward_scale = -200.0  
    tracking_contacts_shaped_force_reward_scale = 1.0  
    
    # --- Baseline Rewards (v10) ---
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    action_rate_reward_scale = -0.0001
    raibert_heuristic_reward_scale = -20.0
    base_level_reward_scale = 0.10
    base_height_reward_scale = 0.60
    flat_orientation_reward_scale = 0.0 # Critical for uneven terrain
    
    lin_vel_z_reward_scale = -2.0
    ang_vel_xy_reward_scale = -0.05
    torque_reward_scale = -2.0e-5
    dof_vel_reward_scale = -1.0e-4

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    events: EventCfg = EventCfg()

    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0, 
        damping=0.0,
    )

    Kp, Kd, torque_limits = 20.0, 0.5, 100.0
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_goal")
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_current")
    
    command_lin_vel_x_range = (-1.0, 1.0)
    command_lin_vel_y_range = (-0.05, 0.05)
    command_yaw_rate_range  = (-1.0, 1.0)
    base_height_target, base_height_min = 0.38, 0.20
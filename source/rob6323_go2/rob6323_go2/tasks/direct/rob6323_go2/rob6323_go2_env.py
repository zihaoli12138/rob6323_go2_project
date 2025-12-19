# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

from .rob6323_go2_env_cfg import Rob6323Go2EnvCfg


class Rob6323Go2Env(DirectRLEnv):
    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # actions
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        # store applied torques for torque regularization (anti-hop)
        self._torques = torch.zeros_like(self._actions)

        # commands: [vx, vy, yaw_rate]
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # logging (baseline + your shaping keys + anti-hop keys)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "rew_action_rate",
                "raibert_heuristic",
                "base_level_exp",
                "base_height_exp",
                # anti-hop regularizers
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "torque_l2",
                "dof_vel_l2",
            ]
        }

        # contact termination index (available because _setup_scene ran inside super().__init__)
        self._base_id, _ = self._contact_sensor.find_bodies("base")

        # baseline: action history (num_envs, action_dim, history=3)
        self.last_actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # baseline: torque-level PD control
        self.Kp = torch.tensor([cfg.Kp] * self.cfg.action_space, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.Kd = torch.tensor([cfg.Kd] * self.cfg.action_space, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.torque_limits = cfg.torque_limits

        # baseline: feet ids for Raibert (robot body indexing)
        self._feet_ids = []
        for name in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]:
            ids, _ = self.robot.find_bodies(name)
            self._feet_ids.append(ids[0])

        # baseline: gait phase state
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_indices = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        self.set_debug_vis(self.cfg.debug_vis)

    @property
    def foot_positions_w(self) -> torch.Tensor:
        """Feet positions in world frame. Shape: (num_envs, 4, 3)."""
        return self.robot.data.body_pos_w[:, self._feet_ids]

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()
        self.desired_joint_pos = self.cfg.action_scale * self._actions + self.robot.data.default_joint_pos

    def _apply_action(self) -> None:
        torques = self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos) - self.Kd * self.robot.data.joint_vel
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)

        # save torques for regularization
        self._torques[:] = torques

        self.robot.set_joint_effort_target(torques)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            [
                t
                for t in (
                    self.robot.data.root_lin_vel_b,
                    self.robot.data.root_ang_vel_b,
                    self.robot.data.projected_gravity_b,
                    self._commands,
                    self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                    self.robot.data.joint_vel,
                    self._actions,
                    self.clock_inputs,
                )
                if t is not None
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # tracking
        lin_vel_error = torch.sum((self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]) ** 2, dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        yaw_rate_error = (self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) ** 2
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        # baseline: action smoothness (1st + 2nd derivative)
        rew_action_rate = torch.sum((self._actions - self.last_actions[:, :, 0]) ** 2, dim=1) * (self.cfg.action_scale**2)
        rew_action_rate += torch.sum(
            (self._actions - 2.0 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]) ** 2, dim=1
        ) * (self.cfg.action_scale**2)

        self.last_actions = torch.roll(self.last_actions, shifts=1, dims=2)
        self.last_actions[:, :, 0] = self._actions

        # baseline: gait + Raibert
        self._step_contact_targets()
        rew_raibert_heuristic = self._reward_raibert_heuristic()

        # your shaping: base level + base height (exp)
        gravity_xy_sq = torch.sum(self.robot.data.projected_gravity_b[:, :2] ** 2, dim=1)
        base_level_mapped = torch.exp(-gravity_xy_sq / self.cfg.base_level_exp_denom)

        base_height = self.robot.data.root_pos_w[:, 2]
        height_err_sq = (base_height - self.cfg.base_height_target) ** 2
        height_mapped = torch.exp(-height_err_sq / self.cfg.base_height_exp_denom)

        # ---------------- anti-hop regularizers ----------------
        # hopping shows up as large vertical base velocity and base rocking (roll/pitch rates)
        lin_vel_z_l2 = self.robot.data.root_lin_vel_b[:, 2] ** 2
        ang_vel_xy_l2 = torch.sum(self.robot.data.root_ang_vel_b[:, :2] ** 2, dim=1)
        torque_l2 = torch.sum(self._torques ** 2, dim=1)
        dof_vel_l2 = torch.sum(self.robot.data.joint_vel ** 2, dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "rew_action_rate": rew_action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "raibert_heuristic": rew_raibert_heuristic * self.cfg.raibert_heuristic_reward_scale * self.step_dt,
            "base_level_exp": base_level_mapped * self.cfg.base_level_reward_scale * self.step_dt,
            "base_height_exp": height_mapped * self.cfg.base_height_reward_scale * self.step_dt,
            # anti-hop (all are costs => cfg scales should be negative)
            "lin_vel_z_l2": lin_vel_z_l2 * self.cfg.lin_vel_z_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_xy_l2 * self.cfg.ang_vel_xy_reward_scale * self.step_dt,
            "torque_l2": torque_l2 * self.cfg.torque_reward_scale * self.step_dt,
            "dof_vel_l2": dof_vel_l2 * self.cfg.dof_vel_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for k, v in rewards.items():
            self._episode_sums[k] += v
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        base_hit = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)

        upside_down = self.robot.data.projected_gravity_b[:, 2] > 0.0

        base_height = self.robot.data.root_pos_w[:, 2]
        too_low = base_height < self.cfg.base_height_min

        died = base_hit | upside_down | too_low
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._torques[env_ids] = 0.0

        # your v1 command ranges
        self._commands[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.command_lin_vel_x_range)
        self._commands[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.command_lin_vel_y_range)
        self._commands[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.command_yaw_rate_range)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        extras = {}
        for k in self._episode_sums.keys():
            extras["Episode_Reward/" + k] = torch.mean(self._episode_sums[k][env_ids]) / self.max_episode_length_s
            self._episode_sums[k][env_ids] = 0.0

        self.extras["log"] = dict(extras)

        self.last_actions[env_ids] = 0.0
        self.gait_indices[env_ids] = 0.0
        self.clock_inputs[env_ids] = 0.0
        self.desired_contact_states[env_ids] = 0.0

    # ---------------- debug viz ----------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        vel_des_scale, vel_des_quat = self._resolve_xy_velocity_to_arrow(self._commands[:, :2])
        vel_scale, vel_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])

        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_quat, vel_des_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_quat, vel_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_vel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_vel.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_vel, dim=1) * 3.0

        heading = torch.atan2(xy_vel[:, 1], xy_vel[:, 0])
        zeros = torch.zeros_like(heading)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading)

        arrow_quat = math_utils.quat_mul(self.robot.data.root_quat_w, arrow_quat)
        return arrow_scale, arrow_quat

    # ---------------- baseline: gait + Raibert ----------------
    def _step_contact_targets(self):
        frequencies = 3.0
        phases = 0.5
        durations = 0.5 * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)

        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)

        foot_indices = [
            self.gait_indices + phases,
            self.gait_indices,
            self.gait_indices,
            self.gait_indices + phases,
        ]
        self.foot_indices = torch.remainder(torch.cat([fi.unsqueeze(1) for fi in foot_indices], dim=1), 1.0)

        for idxs in foot_indices:
            stance = torch.remainder(idxs, 1.0) < durations
            swing = torch.remainder(idxs, 1.0) > durations
            idxs[stance] = torch.remainder(idxs[stance], 1.0) * (0.5 / durations[stance])
            idxs[swing] = 0.5 + (torch.remainder(idxs[swing], 1.0) - durations[swing]) * (0.5 / (1.0 - durations[swing]))

        self.clock_inputs[:, 0] = torch.sin(2.0 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2.0 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2.0 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2.0 * np.pi * foot_indices[3])

    def _reward_raibert_heuristic(self):
        cur_footsteps = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
        footsteps_b = torch.zeros(self.num_envs, 4, 3, device=self.device)

        yaw_inv = math_utils.quat_conjugate(self.robot.data.root_quat_w)
        for i in range(4):
            footsteps_b[:, i, :] = math_utils.quat_apply_yaw(yaw_inv, cur_footsteps[:, i, :])

        desired_stance_width = 0.25
        desired_ys_nom = torch.tensor(
            [desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2],
            device=self.device,
        ).unsqueeze(0)

        desired_stance_length = 0.45
        desired_xs_nom = torch.tensor(
            [desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2],
            device=self.device,
        ).unsqueeze(0)

        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) - 0.5
        frequencies = torch.tensor([3.0], device=self.device)

        x_vel_des = self._commands[:, 0:1]
        yaw_vel_des = self._commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2.0

        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1.0
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys = desired_ys_nom + desired_ys_offset
        desired_xs = desired_xs_nom + desired_xs_offset

        desired = torch.cat((desired_xs.unsqueeze(2), desired_ys.unsqueeze(2)), dim=2)
        err = desired - footsteps_b[:, :, 0:2]
        return torch.sum(err**2, dim=(1, 2))

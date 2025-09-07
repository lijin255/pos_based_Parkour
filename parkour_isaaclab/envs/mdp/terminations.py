# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi
from parkour_isaaclab.envs.mdp import ParkourEvent
 
if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv

def terminate_episode(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.5,
    velocity_threshold: float = 0.15,
    min_episode_length: int = 10,
):  
    reset_buf = torch.zeros((env.num_envs, ), dtype=torch.bool, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_state_w[:,3:7])
    roll_cutoff = torch.abs(wrap_to_pi(roll)) > 1.5
    pitch_cutoff = torch.abs(wrap_to_pi(pitch)) > 1.5
    time_out_buf = env.episode_length_buf >= env.max_episode_length
    parkour_event: ParkourEvent =  env.parkour_manager.get_term('base_parkour')    
    
    # 检查是否到达所有目标点
    all_goals_reached = parkour_event.cur_goal_idx >= env.scene.terrain.cfg.terrain_generator.num_goals
    
    # 只有当到达所有目标点时才检查距离和稳定性条件
    goal_completion_termination = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if torch.any(all_goals_reached) and torch.any(env.episode_length_buf >= min_episode_length):
        # 获取当前机器人位置和最后一个目标点的距离
        robot_pos = asset.data.root_state_w[:, :3]
        current_goal_pos = parkour_event.goal_pos_world[torch.arange(env.num_envs), parkour_event.cur_goal_idx - 1]
        distance_to_final_goal = torch.norm(robot_pos - current_goal_pos, dim=1)
        distance_condition = distance_to_final_goal <= threshold
        
        # 检查机器人身体是否稳定静止
        linear_vel = torch.norm(asset.data.root_lin_vel_w, dim=1)
        angular_vel = torch.norm(asset.data.root_ang_vel_w, dim=1)
        stability_condition = (linear_vel < velocity_threshold) & (angular_vel < 0.2)
        
        # 只有当到达所有目标点且满足距离和稳定性条件时才终止
        goal_completion_termination = all_goals_reached & distance_condition & stability_condition
    
    height_cutoff = asset.data.root_state_w[:, 2] < -0.25
    time_out_buf |= all_goals_reached & (~goal_completion_termination)  # 到达目标但不满足稳定条件时不立即终止
    reset_buf |= time_out_buf
    reset_buf |= roll_cutoff
    reset_buf |= pitch_cutoff
    reset_buf |= height_cutoff
    reset_buf |= goal_completion_termination  # 满足所有条件时终止
    return reset_buf

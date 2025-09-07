from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_apply
from parkour_isaaclab.envs.mdp.parkours import ParkourEvent 
from collections.abc import Sequence

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

import cv2
import numpy as np 

class reward_feet_edge(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]
        self.parkour_event: ParkourEvent =  env.parkour_manager.get_term(cfg.params["parkour_name"])
        self.body_id = self.contact_sensor.find_bodies('base')[0]
        self.horizontal_scale = env.scene.terrain.cfg.terrain_generator.horizontal_scale
        size_x, size_y = env.scene.terrain.cfg.terrain_generator.size
        self.rows_offset = (size_x * env.scene.terrain.cfg.terrain_generator.num_rows/2)
        self.cols_offset = (size_y * env.scene.terrain.cfg.terrain_generator.num_cols/2)
        total_x_edge_maskes = torch.from_numpy(self.parkour_event.terrain.terrain_generator_class.x_edge_maskes).to(device = self.device)
        self.x_edge_masks_tensor = total_x_edge_maskes.permute(0, 2, 1, 3).reshape(
            env.scene.terrain.terrain_generator_class.total_width_pixels, env.scene.terrain.terrain_generator_class.total_length_pixels
        )

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        parkour_name: str,
        ) -> torch.Tensor:
        feet_pos_x = ((self.asset.data.body_state_w[:, self.asset_cfg.body_ids ,0] + self.rows_offset)
                      /self.horizontal_scale).round().long() 
        feet_pos_y = ((self.asset.data.body_state_w[:, self.asset_cfg.body_ids ,1] + self.cols_offset)
                      /self.horizontal_scale).round().long() 
        feet_pos_x = torch.clip(feet_pos_x, 0, self.x_edge_masks_tensor.shape[0]-1)
        feet_pos_y = torch.clip(feet_pos_y, 0, self.x_edge_masks_tensor.shape[1]-1)
        feet_at_edge = self.x_edge_masks_tensor[feet_pos_x, feet_pos_y]
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids] #(N, 4, 3)
        previous_contact_forces = self.contact_sensor.data.net_forces_w_history[:, -1, self.sensor_cfg.body_ids] # N, 4, 3
        contact = torch.norm(contact_forces, dim=-1) > 2.
        last_contacts = torch.norm(previous_contact_forces, dim=-1) > 2.
        contact_filt = torch.logical_or(contact, last_contacts) 
        feet_at_edge = contact_filt & feet_at_edge
        rew = (self.parkour_event.terrain.terrain_levels > 3) * torch.sum(feet_at_edge, dim=-1)
        ## This is for debugging to matching index and x_edge_mask
        # origin = self.x_edge_masks_tensor.detach().cpu().numpy().astype(np.uint8) * 255
        # cv2.imshow('origin',origin)
        # origin[feet_pos_x.detach().cpu().numpy(), feet_pos_y.detach().cpu().numpy()] -= 100
        # cv2.imshow('feet_edge',origin)
        # cv2.waitKey(1)
        return rew

def reward_torques(
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)

def reward_dof_error(    
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)

def reward_hip_pos(
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] \
                                    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)

def reward_ang_vel_xy(
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:,:2]), dim=1)

class reward_action_rate(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_actions = torch.zeros(env.num_envs, 2,  asset.num_joints, dtype= torch.float ,device=self.device)
        
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous_actions[env_ids, 0,:] = 0.
        self.previous_actions[env_ids, 1,:] = 0.

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        self.previous_actions[:, 0, :] = self.previous_actions[:, 1, :]
        self.previous_actions[:, 1, :] = env.action_manager.get_term('joint_pos').raw_actions
        return torch.norm(self.previous_actions[:, 1, :] - self.previous_actions[:,0,:], dim=1)
    
class reward_dof_acc(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_joint_vel = torch.zeros(env.num_envs, 2,  asset.num_joints, dtype= torch.float ,device=self.device)
        self.dt = env.cfg.decimation * env.cfg.sim.dt

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous_joint_vel[env_ids, 0,:] = 0.
        self.previous_joint_vel[env_ids, 1,:] = 0.

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        self.previous_joint_vel[:, 0, :] = self.previous_joint_vel[:, 1, :]
        self.previous_joint_vel[:, 1, :] = asset.data.joint_vel
        return torch.sum(torch.square((self.previous_joint_vel[:, 1, :] - self.previous_joint_vel[:,0,:]) / self.dt), dim=1)
        
def reward_lin_vel_z(
    env: ParkourManagerBasedRLEnv,        
    parkour_name:str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    parkour_event: ParkourEvent =  env.parkour_manager.get_term(parkour_name)
    terrain_names = parkour_event.env_per_terrain_name
    asset: Articulation = env.scene[asset_cfg.name]
    rew = torch.square(asset.data.root_lin_vel_b[:, 2])
    rew[(terrain_names !='parkour_flat')[:,-1]] *= 0.5
    return rew

def reward_orientation(
    env: ParkourManagerBasedRLEnv,   
    parkour_name:str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    parkour_event: ParkourEvent =  env.parkour_manager.get_term(parkour_name)
    terrain_names = parkour_event.env_per_terrain_name
    asset: Articulation = env.scene[asset_cfg.name]
    rew = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    rew[(terrain_names !='parkour_flat')[:,-1]] = 0.
    return rew

def reward_feet_stumble(
    env: ParkourManagerBasedRLEnv,        
    sensor_cfg: SceneEntityCfg ,
    ) -> torch.Tensor: 
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:,0,sensor_cfg.body_ids]
    rew = torch.any(torch.norm(net_contact_forces[:, :, :2], dim=2) >\
            4 *torch.abs(net_contact_forces[:, :, 2]), dim=1)
    return rew.float()

# def reward_tracking_goal_vel(
#     env: ParkourManagerBasedRLEnv, 
#     parkour_name : str, 
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     ) -> torch.Tensor:
#     asset: Articulation = env.scene[asset_cfg.name]
#     parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)
#     target_pos_rel = parkour_event.target_pos_rel
#     target_vel = target_pos_rel / (torch.norm(target_pos_rel, dim=-1, keepdim=True) + 1e-5)
#     cur_vel = asset.data.root_vel_w[:, :2]
#     proj_vel = torch.sum(target_vel * cur_vel, dim=-1)
#     command_vel = env.command_manager.get_command('base_velocity')[:, 0]
#     rew_move = torch.minimum(proj_vel, command_vel) / (command_vel + 1e-5)
#     return rew_move

def reward_tracking_yaw(     
    env: ParkourManagerBasedRLEnv, 
    parkour_name : str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    parkour_event: ParkourEvent =  env.parkour_manager.get_term(parkour_name)
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.root_quat_w
    yaw = torch.atan2(2*(q[:,0]*q[:,3] + q[:,1]*q[:,2]),
                    1 - 2*(q[:,2]**2 + q[:,3]**2))
    return torch.exp(-torch.abs((parkour_event.target_yaw - yaw)))

class reward_delta_torques(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_torque = torch.zeros(env.num_envs, 2,  self.asset.num_joints, dtype= torch.float ,device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous_torque[env_ids, 0,:] = 0.
        self.previous_torque[env_ids, 1,:] = 0.

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        self.previous_torque[:, 0, :] = self.previous_torque[:, 1, :]
        self.previous_torque[:, 1, :] = self.asset.data.applied_torque
        return torch.sum(torch.square((self.previous_torque[:, 1, :] - self.previous_torque[:,0,:])), dim=1)

def reward_collision(
    env: ParkourManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg ,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:,0,sensor_cfg.body_ids]
    return torch.sum(1.*(torch.norm(net_contact_forces, dim=-1) > 0.1), dim=1)

# -------------------positon_reward-------------------------
class GoalProgressTracker(ManagerTermBase):
    """
    用于获取机器人跑酷任务中每个环境的目标点位置信息的类
    """
    
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        """
        初始化目标位置跟踪器
        
        Args:
            cfg: 奖励项配置
            env: 跑酷环境实例
        """
        super().__init__(cfg, env)
        
        # 获取相关组件
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.parkour_event: ParkourEvent = env.parkour_manager.get_term('base_parkour')
        
        # 初始化数据
        self.current_goal_position = self.get_current_goal_position()
        self.reached_goals_count = self.get_reached_goals_count()
        self.current_goal_distance = self.get_current_goal_distance()

        
    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        parkour_name: str,
    ) -> torch.Tensor:
        """
        调用三个方法并打印出值，返回1
        """
        # 调用三个方法获取最新数据
        current_goal_pos = self.get_current_goal_position()
        reached_count = self.get_reached_goals_count()
        goal_distance = self.get_current_goal_distance()
        base_reward = reached_count * 0.2
        # 基于距离计算奖励：距离越小奖励越大，无限接近时奖励为1
        change_reward = 0.2*torch.exp(-goal_distance) 
        # change_reward 
        # 打印前四个环境的信息
        # num_envs_to_print = min(4, env.num_envs)
        # if num_envs_to_print > 0:
        #     print("=" * 60)
        #     print("=== GoalProgressTracker Info (First 4 Envs) ===")
        #     for i in range(num_envs_to_print):
        #         print(f"--- Environment {i} ---")
        #         print(f"Current Goal Position: [{current_goal_pos[i, 0].item():.3f}, {current_goal_pos[i, 1].item():.3f}, {current_goal_pos[i, 2].item():.3f}]")
        #         print(f"Reached Goals Count: {reached_count[i].item()}")
        #         print(f"Current Goal Distance: {goal_distance[i].item():.3f}m")
        #         print(f"Distance Reward: {change_reward[i].item():.4f}")
        #         print()
        #     print("=" * 60)
        # print("reward shape",change_reward.shape)
        return base_reward + change_reward
    
    def get_current_goal_position(self) -> torch.Tensor:
        """
        获得当前目标点的坐标
        
        Returns:
            当前目标点的世界坐标 (x, y, z)
        """
        current_goals_local = self.parkour_event.cur_goals  # 相对于环境原点的位置
        env_origins = self.parkour_event.env_origins  # 环境原点的世界坐标
        
        # 转换为世界坐标
        current_goals_world = torch.zeros(self.num_envs, 3, device=self.device)
        current_goals_world[:, :2] = current_goals_local[:, :2] + env_origins[:, :2]
        current_goals_world[:, 2] = current_goals_local[:, 2]  # 高度信息
        
        return current_goals_world
    
    def get_reached_goals_count(self) -> torch.Tensor:
        """
        获得当前已经达到的目标数
        
        Returns:
            已达到的目标点数量张量
        """
        total_goals = self.parkour_event.num_goals
        current_goal_idx = self.parkour_event.cur_goal_idx
        
        # 如果当前目标索引 >= 总目标数，说明已经完成所有目标
        reach_goal_cutoff = current_goal_idx >= total_goals
        
        # 已达到的目标数 = min(当前目标索引, 总目标数)
        reached_count = torch.where(reach_goal_cutoff, 
                                  torch.full_like(current_goal_idx, total_goals), 
                                  current_goal_idx)
        
        return reached_count.float()
    
    def get_current_goal_distance(self) -> torch.Tensor:
        """
        获得当前目标点和机器人的相对距离
        
        Returns:
            当前目标点和机器人的距离张量
        """
        # 获取机器人当前位置（世界坐标系）
        robot_pos_w = self.asset.data.root_pos_w[:, :3]
        # 获取当前目标点的世界坐标
        current_goal_pos = self.get_current_goal_position()
        relative_distance = torch.norm(current_goal_pos[:, :3] - robot_pos_w[:, :3], dim=1)
        
        return relative_distance
    
def far_but_stop(  
    env:ParkourManagerBasedRLEnv,  
    command_name: str,  
    dist_threshold: float = 0.5,  # 距离阈值，超过此值认为"较远"  
    vel_threshold: float = 0.1,   # 速度阈值，低于此值认为"很小"  
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  
) -> torch.Tensor:  
    """当机器人离目标较远且正向速度很小时提供-1惩罚，其他情况为0"""  
      
    # 提取相关实体和命令  
    asset: Articulation = env.scene[asset_cfg.name]  
    command = env.command_manager.get_command(command_name)  
      
    # 计算到目标的距离  
    target_pos = command[:, :2]  # 三维目标位置  
    pos_dist = torch.norm(target_pos, dim=1)  
      
    # 获取机器人正向速度  
    forward_vel = asset.data.root_lin_vel_b[:, 0]  
    # print("forward velocity:", forward_vel)
    # 判断条件  
    far_from_target = pos_dist > dist_threshold  
    slow_motion = forward_vel < vel_threshold  
      
    # 同时满足两个条件时给予-1惩罚，否则为0  
    penalty_condition = far_from_target & slow_motion  
      
    penalty = torch.where(  
        penalty_condition,  
        torch.full_like(pos_dist, -1.0),  # 恒定-1惩罚  
        torch.zeros_like(pos_dist)        # 其他情况为0  
    )  
      
    return penalty

def reward_distance(
    env: ParkourManagerBasedRLEnv,
    command_name: str,
    Tr: float = 2.0,  # 开始计算奖励的时间阈值（秒）
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    基于公式的任务奖励函数：
    r_task = {
        1/Tr * 1/(1 + ||xb - xb*||^2),  if t > T - Tr
        0,                              otherwise
    }
    
    Args:
        env: 环境实例
        command_name: 命令名称
        Tr: 开始计算奖励的时间阈值（秒）
        asset_cfg: 机器人配置
        
    Returns:
        奖励张量
    """
    # 获取相关实体和命令
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # 计算当前时间 t（从episode开始算起，单位：秒）
    dt = env.cfg.decimation * env.cfg.sim.dt  # 每个控制步的时间
    current_time = env.episode_length_buf * dt  # 当前episode时间
    
    # 计算最大episode时间 T（秒）
    max_episode_time = env.max_episode_length * dt
    
    # 判断是否满足时间条件：t > T - Tr
    time_condition = current_time > (max_episode_time - Tr)
    
    # 计算 ||xb - xb*||^2，使用command的norm来表示机器人当前位置与目标位置的差
    # command[:, :3] 通常包含目标位置相对于当前位置的差值
    position_error_norm = torch.norm(command[:, :3], dim=1)  # ||xb - xb*||
    position_error_squared = position_error_norm ** 2  # ||xb - xb*||^2
    reward_value = (1.0 / Tr) * (1.0 / (1.0 + position_error_squared))
    # 只在满足时间条件时给予奖励，否则为0
    reward = torch.where(
        time_condition,
        reward_value,
        torch.zeros_like(reward_value)
    )
    # print("reward_task_formula:", reward)
    return reward

def stand_still_without_cmd(
    env:ParkourManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚机器人在没有命令的情况下运动."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle), dim=1)
    command = env.command_manager.get_command(command_name)
    pose_command = command[:, :3]
    reward *= torch.norm(pose_command, dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

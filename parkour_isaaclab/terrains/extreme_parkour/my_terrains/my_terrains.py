from __future__ import annotations

import numpy as np
import scipy.spatial.transform as tf
import torch
import trimesh
from typing import TYPE_CHECKING
from scipy import interpolate

from isaaclab.terrains.trimesh.utils import *  # noqa: F401, F403
from isaaclab.terrains.trimesh.utils import make_border, make_plane
if TYPE_CHECKING:
    from .my_terrains_cfg import CenterPlatformCfg,LargeGapCfg, ExtremeParkourRoughTerrainCfg
    from ...parkour_terrain_generator_cfg import ParkourSubTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh 
from ...utils import parkour_field_to_mesh 
def padding_height_field_raw(
    height_field_raw: np.ndarray, 
    cfg: ExtremeParkourRoughTerrainCfg
    ) -> np.ndarray:
    pad_width = int(cfg.pad_width // cfg.horizontal_scale)
    pad_height = int(cfg.pad_height // cfg.vertical_scale)
    height_field_raw[:, :pad_width] = pad_height
    height_field_raw[:, -pad_width:] = pad_height
    height_field_raw[:pad_width, :] = pad_height
    height_field_raw[-pad_width:, :] = pad_height
    height_field_raw = np.rint(height_field_raw).astype(np.int16)
    return height_field_raw

def random_uniform_terrain(
    difficulty: float, 
    cfg: ExtremeParkourRoughTerrainCfg,
    height_field_raw: np.ndarray,
    ):
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # # -- downsampled scale
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
    # -- height
    max_height = (cfg.noise_range[1] - cfg.noise_range[0]) * difficulty + cfg.noise_range[0]
    height_min = int(-cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(max_height / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)

    # create range of heights possible
    height_range = np.arange(height_min, height_max + height_step, height_step)
    # sample heights randomly from the range along a grid
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))
    # create interpolation function for the sampled heights
    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)
    # interpolate the sampled heights to obtain the height field
    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    # round off the interpolated heights to the nearest vertical step
    z_upsampled = np.rint(z_upsampled).astype(np.int16)
    height_field_raw += z_upsampled 
    return height_field_raw 
@parkour_field_to_mesh
def centerplatform_terrain(    
    difficulty: float, cfg: CenterPlatformCfg ,num_goals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:    
    # 生成高度场数组    
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)    
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)    
    offset = int(3.0 / cfg.horizontal_scale)
    # 创建基础高度场（全零）    
    height_field = np.zeros((width_pixels, length_pixels), dtype=np.int16)    
        
    # 计算平台区域    
    platform_pixels = int(cfg.platform_width / cfg.horizontal_scale)    
    center_x = width_pixels // 2    #向下取整的整数
    center_y = (length_pixels +offset)  // 2    
        
    # 设置中间平台高度    
    platform_height_pixels = int(2*(cfg.platform_height + 0.30*difficulty) / cfg.vertical_scale)    
    x1 = center_x - platform_pixels // 2    
    x2 = center_x + platform_pixels // 2    
    y1 = center_y - platform_pixels // 2    
    y2 = center_y + platform_pixels // 2      
    height_field[x1:x2, y1:y2] = platform_height_pixels#在坐标范围内设置高度
    fr_platform_height_pixels = int((cfg.platform_height + 0.30*difficulty) / cfg.vertical_scale)    
    fr_y1 = y1 - platform_pixels
    height_field[x1:x2, fr_y1:y1] = fr_platform_height_pixels
    
    # 使用 ExtremeParkourRoughTerrainCfg 的粗糙度逻辑替换原有噪声
    if cfg.apply_roughness:
        height_field = random_uniform_terrain(difficulty, cfg, height_field)
    
    # 添加边界处理
    height_field = padding_height_field_raw(height_field, cfg)
    
    # 生成目标点 - 在平台上随机分布
    goals = np.zeros((num_goals, 2))
    goal_heights = np.zeros(num_goals, dtype=np.int16)
    
    for i in range(num_goals):
        # 在平台区域内随机生成目标点
        goal_x = np.random.randint(x1, x2)
        goal_y = np.random.randint(y1, y2)
        goals[i] = [goal_x * cfg.horizontal_scale, goal_y * cfg.horizontal_scale]
        goal_heights[i] = height_field[goal_x, goal_y]
    
    return height_field, goals, goal_heights
@parkour_field_to_mesh
def gap_terrain(  
    difficulty: float, cfg: LargeGapCfg,num_goals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  
    from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh  
      
    gap_width = cfg.gap[0] + difficulty * (cfg.gap[1]-cfg.gap[0])  
    print("difficulty", difficulty, "gap_width", gap_width)  
      
    # 生成高度场数组  
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)  
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)  
      
    # 创建基础高度场（全零）  
    base_height_pixels = int(-1.0 / cfg.vertical_scale)  # -1米基础高度  
    height_field = np.full((width_pixels, length_pixels), base_height_pixels, dtype=np.int16)   
      
    # 计算平台区域（边缘平台）  
    platform_pixels = int(cfg.platform_size / cfg.horizontal_scale)  
    offset = int(2.0 / cfg.horizontal_scale)  
    platform_height_pixels = int(cfg.platform_height / cfg.vertical_scale)  
      
    # 设置边缘平台高度  从中间算起
    height_field[:, :platform_pixels+offset] = platform_height_pixels  # 上边缘    
    height_field[:, -platform_pixels:] = platform_height_pixels  # 下边缘 
      
    # 计算小平台参数     
    gap_pixels = int(gap_width / cfg.horizontal_scale)  
  
 
    # 计算长方形平台参数  
    platform_width_pixels = int(cfg.platform_width / cfg.horizontal_scale)
    platform_length_pixels = int(cfg.platform_length / cfg.horizontal_scale)
    gap_pixels = int(gap_width / cfg.horizontal_scale)
    x_center = width_pixels // 2  # x方向中心点
    x1 = x_center - platform_length_pixels // 2
    x2 = x_center + platform_length_pixels // 2
    
    # 记录平台位置用于生成目标点
    platform_positions = []
    
    # 沿y方向布置平台
    current_y = platform_pixels+offset
    while True:
        # 计算下一个标准平台的位置
        next_y1 = current_y
        next_y2 = next_y1 + platform_width_pixels

        # 检查下一个标准平台是否会越界
        if next_y2 > length_pixels - platform_pixels:
            # 如果会越界，则停止放置标准平台
            break
        
        # 如果不会越界，正常放置平台
        height_field[x1:x2, next_y1:next_y2] = platform_height_pixels
        # 记录平台中心位置
        platform_center_y = (next_y1 + next_y2) // 2
        platform_positions.append(platform_center_y)
        
        # 更新到下一个平台的起始位置
        current_y = next_y2 + gap_pixels

    # 循环结束后，检查是否可以在剩余空间放置一个与边缘相接的平台
    if current_y < length_pixels - platform_pixels:
        y1 = current_y
        y2 = length_pixels - platform_pixels
        height_field[x1:x2, y1:y2] = platform_height_pixels
        # 记录最后一个平台的中心位置
        platform_center_y = (y1 + y2) // 2
        platform_positions.append(platform_center_y)

    # 使用 ExtremeParkourRoughTerrainCfg 的粗糙度逻辑替换原有噪声
    if cfg.apply_roughness:
        height_field = random_uniform_terrain(difficulty, cfg, height_field)
    
    # 生成目标点 - 在平台中心和平台间隙中设置目标点
    goals = np.zeros((num_goals, 2))  # 二维坐标 [x, y]
    goal_heights = np.zeros(num_goals)  # 单独存储高度
    
    # 添加起始点（在起始边缘平台上）
    # start_y = platform_pixels // 2  # 起始平台的中心
    platform_height_raw = round(0.3 / cfg.vertical_scale)  # 0.3米转换为像素高度
    # goals[0] = [x_center, start_y]  # 使用像素坐标
    # goal_heights[0] = platform_height_raw
    
    goal_index = 0
    
    # 在每个中间平台的中心设置目标点
    for i, platform_y in enumerate(platform_positions):
        if goal_index >= num_goals:
            break
        # 平台中心点高度固定为0.3（转换为像素）
        goals[goal_index] = [x_center, platform_y]
        goal_heights[goal_index] = platform_height_raw
        goal_index += 1
        
        # 在当前平台和下一个平台之间的间隙中设置目标点
        if i < len(platform_positions) - 1 and goal_index < num_goals:
            next_platform_y = platform_positions[i + 1]
            gap_center_y = (platform_y + next_platform_y) // 2
            # 间隙中心点高度为间隙值的一半（转换为像素）
            gap_height_raw = round((gap_width / 4.0) / cfg.vertical_scale) + platform_height_raw
            goals[goal_index] = [x_center, gap_center_y]
            goal_heights[goal_index] = gap_height_raw
            goal_index += 1
    
    # 如果还有剩余的目标点槽位，将所有剩余目标点都放在最后一个放置的位置
    if goal_index > 0:  # 确保至少有一个目标点已经放置
        last_goal_x = goals[goal_index - 1][0]
        last_goal_y = goals[goal_index - 1][1]
        last_goal_height = goal_heights[goal_index - 1]
        
        while goal_index < num_goals:
            goals[goal_index] = [last_goal_x, last_goal_y]
            goal_heights[goal_index] = last_goal_height
            goal_index += 1
    print("goal_heights",goal_heights)
    # 添加边界处理
    height_field = padding_height_field_raw(height_field, cfg)     
    #goal_heights不能返回像素，不然会被截断
    return height_field, goals * cfg.horizontal_scale, goal_heights 
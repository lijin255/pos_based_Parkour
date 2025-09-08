import warnings
from dataclasses import MISSING
from typing import Literal, Tuple

import isaaclab.terrains.trimesh.mesh_terrains as mesh_terrains
import isaaclab.terrains.trimesh.utils as mesh_utils_terrains
from isaaclab.utils import configclass

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from .my_terrains import centerplatform_terrain,gap_terrain
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from ...parkour_terrain_generator_cfg import ParkourSubTerrainBaseCfg

@configclass
class ExtremeParkourRoughTerrainCfg(ParkourSubTerrainBaseCfg):
    apply_roughness: bool = True 
    apply_flat: bool = False 
    downsampled_scale: float | None = 0.075
    noise_range: tuple[float,float] = (0.02, 0.06)
    noise_step: float = 0.005
    x_range: tuple[float, float] = (0.8, 1.5)
    y_range: tuple[float, float] = (-0.4, 0.4)
    half_valid_width: tuple[float, float] = (0.6, 1.2)
    pad_width: float = 0.1 
    pad_height: float = 0.0
@configclass
class CenterPlatformCfg(ExtremeParkourRoughTerrainCfg): 
    function = centerplatform_terrain 
    platform_width: float = 1.0  
    platform_height: float = 0.3  
    num_goals: int = 8
@configclass  
class LargeGapCfg(ExtremeParkourRoughTerrainCfg):  
    function = gap_terrain  
    gap: tuple[float, float] = (0.3, 1.0)  # 间隙宽度范围 
    platform_size: float = 1.0  
    platform_width: float = 1.0  
    platform_length: float = 1.5  
    platform_height: float = 0.0  
    apply_roughness :bool = True
    noise_range: tuple[float, float] = (0.005, 0.05)  # 噪声范围  
    noise_step: float = 0.005  # 噪声步长
    num_goals: int = 8
    
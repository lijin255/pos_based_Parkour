from isaaclab.utils import configclass
##
# Pre-defined configs
##
from parkour_isaaclab.terrains.extreme_parkour.extreme_parkour_terrains_cfg import ExtremeParkourRoughTerrainCfg
  # isort: skip
from parkour_isaaclab.envs import ParkourManagerBasedRLEnvCfg
from .parkour_mdp_cfg import * 
from parkour_tasks.default_cfg import  CAMERA_USD_CFG, CAMERA_CFG, VIEWER
from .parkour_teacher_cfg import ParkourTeacherSceneCfg
@configclass
class ParkourStudentSceneCfg(ParkourTeacherSceneCfg):
    depth_camera = CAMERA_CFG
    depth_camera_usd = None
    
    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator.num_rows = 10
        self.terrain.terrain_generator.num_cols = 20
        self.terrain.terrain_generator.horizontal_scale = 0.1
        
        for key, sub_terrain in self.terrain.terrain_generator.sub_terrains.items():
            sub_terrain: ExtremeParkourRoughTerrainCfg
            sub_terrain.use_simplified = True 
            if key == 'demo':
                sub_terrain.proportion = 0.15
                sub_terrain.y_range = (-0.1, 0.1)

            elif key =='parkour_step':
                sub_terrain.proportion = 0.2
                sub_terrain.y_range = (-0.1, 0.1)

            elif key =='parkour_flat':
                sub_terrain.proportion = 0.05
                sub_terrain.y_range = (-0.1, 0.1)

            elif key =='parkour_gap':
                sub_terrain.proportion = 0.2
                sub_terrain.y_range = (-0.1, 0.1)

            elif key =='parkour_hurdle':
                sub_terrain.proportion = 0.2
                sub_terrain.y_range = (-0.1, 0.1)
            sub_terrain.horizontal_scale = 0.1


@configclass
class UnitreeGo2StudentParkourEnvCfg(ParkourManagerBasedRLEnvCfg):
    viewer = VIEWER 
    scene: ParkourStudentSceneCfg = ParkourStudentSceneCfg(num_envs=77, env_spacing=1.)
    # Basic settings
    observations: StudentObservationsCfg = StudentObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: StudentRewardsCfg = StudentRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    parkours: ParkourEventsCfg = ParkourEventsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**18
        # update sensor update periods
        self.scene.depth_camera.update_period = self.sim.dt * self.decimation
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        self.scene.terrain.terrain_generator.curriculum = True
        self.actions.joint_pos.use_delay = True
        self.actions.joint_pos.history_length = 8

@configclass
class UnitreeGo2StudentParkourEnvCfg_PLAY(UnitreeGo2StudentParkourEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.depth_camera_usd = CAMERA_USD_CFG
        # self.parkours.base_parkour.debug_vis = True
        # self.commands.base_velocity.debug_vis = True
        self.scene.num_envs = 1
        # self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.random_difficulty = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.7,1.0)
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.randomize_rigid_body_com = None
        self.events.randomize_rigid_body_mass = None
        self.events.physics_material = None
        self.events.push_by_setting_velocity = None
        self.events.random_camera_position.params['rot_noise_range'] = {'pitch':(0, 0)}
        # self.commands.base_velocity.resampling_time_range = (60.,60.)
        self.episode_length_s = 60.

        for key, sub_terrain in self.scene.terrain.terrain_generator.sub_terrains.items():
            if key =='parkour_flat':
                sub_terrain.proportion = 0.0
            else:
                sub_terrain.proportion = 0.2
                sub_terrain.noise_range = (0.02, 0.02)

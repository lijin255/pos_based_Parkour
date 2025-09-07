
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
##
# Pre-defined configs
##
from parkour_isaaclab.terrains.extreme_parkour.config.parkour import EXTREME_PARKOUR_TERRAINS_CFG  # isort: skip
from parkour_isaaclab.terrains.extreme_parkour.config.parkour import My_TERRAINS_CFG  # isort: skip
from parkour_isaaclab.envs import ParkourManagerBasedRLEnvCfg
from .parkour_mdp_cfg import * 
from parkour_tasks.default_cfg import ParkourDefaultSceneCfg, VIEWER

@configclass
class ParkourTeacherSceneCfg(ParkourDefaultSceneCfg):
    
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.375, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=[1.65, 1.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", 
                                      history_length=2, 
                                      track_air_time=True, 
                                      debug_vis= False,
                                      force_threshold=1.
                                      )
    def __post_init__(self):
        super().__post_init__()
        # self.terrain.terrain_generator = EXTREME_PARKOUR_TERRAINS_CFG
        self.terrain.terrain_generator = My_TERRAINS_CFG
        
@configclass
class UnitreeGo2TeacherParkourEnvCfg(ParkourManagerBasedRLEnvCfg):
    # viewer = VIEWER 
    scene: ParkourTeacherSceneCfg = ParkourTeacherSceneCfg(num_envs=6144, env_spacing=1.)
    # Basic settings
    observations: TeacherObservationsCfg = TeacherObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: TeacherRewardsCfg = TeacherRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    parkours: ParkourEventsCfg = ParkourEventsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.parkours.base_parkour.debug_vis = True
        self.decimation = 4
        self.episode_length_s = 12.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**18
        # update sensor update periods
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        self.scene.terrain.terrain_generator.curriculum = True
        self.actions.joint_pos.use_delay = False
        self.actions.joint_pos.history_length = 1
        self.events.random_camera_position = None

@configclass
class UnitreeGo2TeacherParkourEnvCfg_PLAY(UnitreeGo2TeacherParkourEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        # self.parkours.base_parkour.debug_vis = True
        # self.commands.base_velocity.debug_vis = True

        self.scene.num_envs = 4
        # spawn the robot randomly in the grid (instead of their terrain levels)
        # self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.random_difficulty = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.0,0.3)
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.randomize_rigid_body_com = None
        self.events.randomize_rigid_body_mass = None
        # self.commands.base_velocity.resampling_time_range = (60.,60.)
        self.episode_length_s = 60.

        # for key, sub_terrain in self.scene.terrain.terrain_generator.sub_terrains.items():
        #     if key =='parkour_flat':
        #         sub_terrain.proportion = 0.0
        #     else:
        #         sub_terrain.proportion = 0.2
        #         sub_terrain.noise_range = (0.02, 0.02)

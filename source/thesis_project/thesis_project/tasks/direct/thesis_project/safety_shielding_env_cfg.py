# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from thesis_project.robots.franka import FRANKA_PANDA_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg, RigidObject
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import os

# URDF_PATH = r"C:\Users\franc\miniconda3\envs\env_isaaclab\Lib\site-packages\isaacsim\extscache\isaacsim.asset.importer.urdf-2.3.10+106.4.0.wx64.r.cp310\data\urdf\robots\franka_description\robots\panda_arm_hand.urdf"
# print("Urdf_path: ", URDF_PATH)
# print("State: ", os.path.exists(URDF_PATH))

@configclass
class SafetyEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10
    # - space definition
    action_space = 7
    observation_space = 41
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1/60, render_interval=decimation)

    urdf_path = r"C:\Users\franc\miniconda3\envs\env_isaaclab\Lib\site-packages\isaacsim\extscache\isaacsim.asset.importer.urdf-2.3.10+106.4.0.wx64.r.cp310\data\urdf\robots\franka_description\robots\panda_arm_hand.urdf"
    table_height = 0.75        # altezza del piano (top) in metri
    table_thickness = 0.05     # spessore del tavolo in metri
    obstacle_height = 0.2
    obstacle_radius = 0.05
    # shielding thresholds
    d_in = 0.05
    d_out = d_in + 0.04
    safe_margin = 0.1
    link_radius =0.05
    clip_dist = 0.1
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.5, table_height),  # piano 1.0 x 0.8, spessore 5cm
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # center_z = top_z - thickness/2
            pos=(0.3, 0.0, table_height/2),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    #robot(s)
    robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.init_state.pos = (0.0, 0.0, table_height)
    # robot_cfg.init_state.joint_pos["panda_joint4"] = -2.350
    # robot_cfg.init_state.joint_pos["panda_joint6"] = 2.500
    robot_cfg.init_state.joint_pos = {
        "panda_joint1": -0.27689978081008815,
        "panda_joint2":  -0.5340115896900678,
        "panda_joint3":  -0.4743214197617522,
        "panda_joint4": -2.930253809731531,
        "panda_joint5": -0.35696643262928796,
        "panda_joint6":  2.413823638632636, 
        "panda_joint7": 0.3639460666111401,
        "panda_finger_joint1": 0.04,
        "panda_finger_joint2": 0.04,
    }
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4, replicate_physics=True)
    dof_names = ["panda_joint1","panda_joint2","panda_joint3","panda_joint4","panda_joint5","panda_joint6","panda_joint7",
                 "panda_finger_joint1", "panda_finger_joint2"]
    
    delta_q = 0.2
    safe_eps_dq = 0.08
    K_p = 50.0
    K_d = 0.1
    goal_success_dist = 0.02
    
    action_scale = 0.7
    dof_velocity_scale = 0.3
    action_penalty_scale = 0.02
    joint_reward_scale = 1.0
    lift_reward_scale = 1.25
    grasp_reward_scale = 3.0
    goal_reward_scale = 2.5
    success_reward_scale = 3.0
    obstacle_penalty_scale = 5.0
    sh_activ_penalty = 0.0 #3.0 con policy principale
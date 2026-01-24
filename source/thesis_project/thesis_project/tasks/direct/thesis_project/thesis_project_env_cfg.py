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

@configclass
class ThesisProjectEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10
    # - space definition
    action_space = 9
    observation_space = 46
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1/60, render_interval=decimation)

    table_height = 0.75        # altezza del piano (top) in metri
    table_thickness = 0.05     # spessore del tavolo in metri
    obstacle_height = 0.2
    obstacle_radius = 0.05
    safe_margin = 0.12
    link_radius =0.04
    clip_dist = 0.1
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 0.8, table_height),  # piano 1.0 x 0.8, spessore 5cm
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
    robot_cfg.init_state.joint_pos["panda_joint4"] = -2.350
    robot_cfg.init_state.joint_pos["panda_joint6"] = 2.500
    #scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=4, replicate_physics=True)
    dof_names = ["panda_joint1","panda_joint2","panda_joint3","panda_joint4","panda_joint5","panda_joint6","panda_joint7",
                 "panda_finger_joint1", "panda_finger_joint2"]
    
    goal_success_dist = 0.02
    
    action_scale = 1.0
    dof_velocity_scale = 0.1
    action_penalty_scale = 0.02
    rot_reward_scale = 1.0
    lift_reward_scale = 1.2
    grasp_reward_scale = 3.0
    goal_reward_scale = 2.5
    success_reward_scale = 3.0
    obstacle_penalty_scale = 10.0
    terminal_collision_penalty = 7.0
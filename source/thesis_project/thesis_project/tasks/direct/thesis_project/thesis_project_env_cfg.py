# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from thesis_project.robots.franka import FRANKA_PANDA_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# @configclass
# class ThesisProjectEnvCfg(DirectRLEnvCfg):
#     # env
#     decimation = 2
#     episode_length_s = 5.0
#     # - spaces definition
#     action_space = 2
#     observation_space = 3
#     state_space = 0
#     # simulation
#     sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
#     # robot(s)
#     robot_cfg: ArticulationCfg = JETBOT_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
#     # scene
#     scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=50, env_spacing=4.0, replicate_physics=True)
#     dof_names = ["left_wheel_joint", "right_wheel_joint"]

@configclass
class ThesisProjectEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10
    # - space definition
    action_space = 9
    observation_space = 33
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1/60, render_interval=decimation)
    #robot(s)
    robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # robot_cfg.init_state.pos = (0.0, 0.0, 1.05)
    # robot_cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)
    #scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=4, replicate_physics=True)
    dof_names = ["panda_joint1","panda_joint2","panda_joint3","panda_joint4","panda_joint5","panda_joint6","panda_joint7",
                 "panda_finger_joint1", "panda_finger_joint2"]
    
    action_scale = 0.8
    dof_velocity_scale = 0.1
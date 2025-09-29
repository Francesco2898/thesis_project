# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import RigidObjectCfg, RigidObject
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from torch.utils.tensorboard import SummaryWriter

from .thesis_project_env_cfg import ThesisProjectEnvCfg

class ThesisProjectEnv(DirectRLEnv):
    cfg: ThesisProjectEnvCfg

    def __init__(self, cfg: ThesisProjectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
            
        self.close_steps_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                # SummaryWriter per TensorBoard
        self.writer = SummaryWriter(log_dir="runs/my_run")
        self.global_step = 0  # contatore manuale degli step

        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        # auxiliary variables
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.effort_limits = self.robot.data.joint_effort_limits
        self.robot_dof_lower_limit = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device = self.device)
        self.robot_dof_upper_limit = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device = self.device)
        self.robot_effort_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        # robot link idxs
        self.hand_link_idx = self.robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self.robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self.robot.find_bodies("panda_rightfinger")[0][0]

        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limit)
        self.robot_dof_speed_scales[self.robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self.robot.find_joints("panda_finger_joint2")[0]] = 0.1

        # target auxiliary variables
        # sample of the new target in a box space
        self.box_min = torch.tensor([-0.3, -0.3, 0.0], device=self.device)
        self.box_max = torch.tensor([0.3, 0.3, 0.5], device=self.device)

    def _setup_scene(self):

        self.robot = Articulation(self.cfg.robot_cfg)
        print("[DEBUG] Actuator config:")
        print(self.robot.cfg.actuators)
        # target position in local coordinates
        self.target_pos = torch.tensor([[0.5, 0.5, 0.5]], device=self.device).repeat(self.num_envs, 1)

        target_obj_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/TargetObject",
            spawn=sim_utils.SphereCfg(
                radius=0.03,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0)
            ),
            debug_vis=True
        )
        self.target_obj = RigidObject(target_obj_cfg)

        # adding a marker following the target
        target_marker_cfg = VisualizationMarkersCfg(
            prim_path="Visuals/target_marker",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
                )
            }
        )
        self.target_marker = VisualizationMarkers(target_marker_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["target"] = self.target_obj
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # # self.robot_effort_targets = self.actions * self.effort_limits * self.cfg.action_scale
        # delta_torque = self.actions * self.effort_limits * self.cfg.action_scale * self.dt
        # self.robot_effort_targets = torch.clamp(
        #     self.robot_effort_targets + delta_torque,
        #     -self.effort_limits,
        #     self.effort_limits
        # )
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limit, self.robot_dof_upper_limit)

    def _apply_action(self) -> None:

        # Torque Control
        # self.robot.set_joint_effort_target(self.robot_effort_targets, joint_ids=self.dof_idx)

        # Position control
        self.robot.set_joint_position_target(self.robot_dof_targets)


    def _get_observations(self) -> dict:
        # end effector state
        ee_pos = self.robot.data.body_pos_w[:, self.hand_link_idx] # world coordinates
        ee_pos_local = self.get_env_local_pos(self.scene.env_origins, ee_pos) # local coordinates
        ee_vel = self.robot.data.body_lin_vel_w[:, self.hand_link_idx] # velocity
        ee_ang = self.robot.data.body_ang_vel_w[:, self.hand_link_idx] # angular velocity

        # joint state
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel

        joint_pos_norm = 2.0 * (joint_pos - self.robot_dof_lower_limit)/(self.robot_dof_upper_limit - self.robot_dof_lower_limit) - 1.0
        joint_vel_norm = joint_vel * self.cfg.dof_velocity_scale

        # target state
        targ_vec = self.target_pos - ee_pos_local # vector distance from the ee
        targ_dist = torch.norm(targ_vec, dim=-1)
        # print("[DEBUG] target dist = ", targ_dist.mean().item())
        obs = torch.hstack((joint_pos_norm,
                            joint_vel_norm,
                            ee_pos_local,
                            ee_vel,
                            ee_ang,
                            targ_vec,
                            self.target_pos
                            ))
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # end effector position in world frame
        ee_pos = self.robot.data.body_pos_w[:, self.hand_link_idx]
        # ee pos in local reference frame
        ee_pos_local = self.get_env_local_pos(self.scene.env_origins, ee_pos)
        # end effector velocity
        ee_vel = self.robot.data.body_lin_vel_w[:, self.hand_link_idx]
        ee_ang = self.robot.data.body_ang_vel_w[:, self.hand_link_idx]
        # target 
        targ_dist = torch.norm(self.target_pos - ee_pos_local, p=2, dim=-1)
        # close = targ_dist < 0.05
        # next_buf = torch.where(close, self.success_steps_buf+1, torch.zeros_like(self.success_steps_buf))
        # just_succeeded = (next_buf == 5)
        # reward_success = just_succeeded.float() * 10.0
        total_reward = self.compute_rewards(targ_dist)
        self.global_step += 1  # aggiorna il contatore step
        if self.global_step % 100 == 0:
            # print("[DEBUG] Logging step:", self.global_step)
            self.writer.add_scalar("Metrics/reward_mean", total_reward.mean().item(), self.global_step)
            self.writer.add_scalar("Metrics/target_dist_mean", targ_dist.mean().item(), self.global_step)
            self.writer.add_scalar("Metrics/target_dist_min", targ_dist.min().item(), self.global_step)
            self.writer.add_scalar("Metrics/target_dist_max", targ_dist.max().item(), self.global_step)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        terminated = self.close_steps_buf >= 5
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        # if torch.any(success):
        #     print(f"[DEBUG] Success envs:", torch.nonzero(success).squeeze(-1).cpu().numpy())
        # if torch.any(time_out):
        #     print(f"[DEBUG] Time-out envs:", torch.nonzero(time_out).squeeze(-1).cpu().numpy())
        # in _get_dones, subito prima di restituire:
        # if not hasattr(self, "success_rate_buf"):
        #     self.success_rate_buf = torch.zeros(self.num_envs, device=self.device)
        # self.success_rate_buf = success.float()  # 1 se ha avuto successo in questo step
        # # poi nel tuo loop di training logga success_rate_buf.mean().item() come "Metrics/success_rate"
        # self.extras["success_rate"] = self.success_rate_buf.mean().item()
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # rtesetting the steps close buffer
        self.close_steps_buf[env_ids] = 0


        # robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self.robot.num_joints),
            device = self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limit, self.robot_dof_upper_limit)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot_dof_targets[env_ids] = joint_pos

        # local position wrt the env
        new_target_pos_local = sample_uniform(self.box_min, self.box_max, (len(env_ids), 3), device=self.device)
        self.target_pos[env_ids] = new_target_pos_local

        new_target_pos_world = new_target_pos_local + self.scene.env_origins[env_ids]
        targ_state = torch.cat((
            new_target_pos_world,
            torch.zeros((len(env_ids),  3), device=self.device), # null linear velocity
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).repeat(len(env_ids), 1),
            torch.zeros((len(env_ids),  3), device=self.device) # null angular velocity
        ), dim=-1)
        # print("[DEBUG] env_ids:", env_ids)
        # print("[DEBUG] targ_state shape:", targ_state.shape)
        # print("[DEBUG] targ_state:", targ_state)
        self.target_obj.write_root_state_to_sim(targ_state, env_ids)
        self.target_marker.visualize((self.target_pos + self.scene.env_origins).cpu())
        # print("[DEBUG] target_obj root state shape:", self.target_obj.data.root_state_w.shape)
        # print("[DEBUG] target pos world: ", self.scene.env_origins + self.target_pos[env_ids])
        # print("[DEBUG] sim root state:", self.target_obj.data.root_state_w[:, :3])

    def get_env_local_pos(self, env_pos: torch.Tensor, world_pos: torch.Tensor, world_quat: torch.Tensor | None = None):
        # compute position in environment local coordinate system
        local_pose = world_pos - env_pos
        if world_quat == None: # interested only in the position
            return local_pose
        else: # interested also in the orientation
            qw, qx, qy, qz = world_quat.unbind(-1)
            return torch.stack([local_pose[:, 0], local_pose[:, 1], local_pose[:, 2], qw, qx, qy, qz],dim=-1)
        
    def compute_rewards(self, targ_dist: torch.Tensor):
        # reward based on the distance from the target
        # r_dist = 1.0 - torch.tanh(targ_dist)
        r_dist = 1.0 / (1.0 + targ_dist**2)
        r_dist *= r_dist
        r_dist = torch.where(targ_dist < 0.02, r_dist*2, r_dist)

        # distance improvement
        if not hasattr(self, "prev_targ_dist"):
            self.prev_targ_dist = targ_dist.clone()
        improvement = torch.clamp(self.prev_targ_dist - targ_dist, min=0.0)
        r_progress = 0.8 * improvement
        self.prev_targ_dist = targ_dist.detach()

        # # success reward
        # self.close_steps_buf = torch.where(
        #     targ_dist < 0.05,
        #     self.close_steps_buf + 1,
        #     torch.zeros_like(self.close_steps_buf)
        # )
        # just_succeeded = (self.close_steps_buf == 5)
        # r_success = just_succeeded.float() * 2.0

        # limits = (self.effort_limits + 1e-8)
        # pen_effect = 1e-4 * torch.sum((self.robot_effort_targets/limits)**2, dim=-1)
        reward = r_dist + r_progress
        return reward
    
    # def compute_rewards(self, targ_dist: torch.Tensor):
    #     # reward based on the distance from the target
    #     comp_exp = 2.0 # compression exponent
    #     r_dist = 1.0/(1.0 + targ_dist**2)
    #     reward_dist = r_dist**comp_exp
    #     # bonus reward if the ee is really close to the target
    #     reward_close = torch.exp(-50 * targ_dist)
    #     # penalization based on the distance from the target
    #     penalization_dist = targ_dist * 0.01

    #     reward = reward_dist + reward_close - penalization_dist
    #     return reward


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward

# def define_markers() -> VisualizationMarkers:
#     """Define markers with various different shapes"""
#     marker_cfg = VisualizationMarkersCfg(
#         prim_path="Visuals/myMarkers",
#         markers={
#             "forward": sim_utils.UsdFileCfg(
#                 usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
#                 scale=(0.25, 0.25, 0.5),
#                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
#             ),
#             "command": sim_utils.UsdFileCfg(
#                 usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
#                 scale=(0.25, 0.25, 0.5),
#                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
#             ),
#         },
#     )
#     return VisualizationMarkers(cfg=marker_cfg)


# class ThesisProjectEnv(DirectRLEnv):
#     cfg: ThesisProjectEnvCfg

#     def __init__(self, cfg: ThesisProjectEnvCfg, render_mode: str | None = None, **kwargs):
#         super().__init__(cfg, render_mode, **kwargs)

#         self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

#     def _setup_scene(self):
#         self.robot = Articulation(self.cfg.robot_cfg)
#         # add ground plane
#         spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
#         # clone and replicate
#         self.scene.clone_environments(copy_from_source=False)
#         # add articulation to scene
#         self.scene.articulations["robot"] = self.robot
#         # add lights
#         light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
#         light_cfg.func("/World/Light", light_cfg)

#         self.visualization_markers = define_markers()

#         # setting aside useful variables for later
#         self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()
#         self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda()
#         self.commands = torch.randn((self.cfg.scene.num_envs, 3)).cuda()
#         self.commands[:,-1] == 0.0
#         self.commands = self.commands/torch.linalg.norm(self.commands, dim = 1, keepdim = True)

#         # offsets to account for atan range and keep things on [-pi, pi]
#         ratio = self.commands[:,1]/(self.commands[:,0]+1E-8)
#         gzero = torch.where(self.commands > 0, True, False)
#         lzero = torch.where(self.commands < 0, True, False)
#         plus = lzero[:,0]*gzero[:,1]
#         minus = lzero[:,0]*lzero[:,1]
#         offsets = torch.pi*plus - torch.pi*minus
#         self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

#         self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
#         self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
#         self.marker_offset[:,-1] = 0.5
#         self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
#         self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()

#     def _visualize_markers(self):
#         # get marker locations and orientations
#         self.marker_locations = self.robot.data.root_pos_w
#         self.forward_marker_orientations = self.robot.data.root_quat_w
#         self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()

#         # offset markers so they are above the jetbot
#         loc = self.marker_locations + self.marker_offset
#         loc = torch.vstack((loc, loc))
#         rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations))

#         # render the markers
#         all_envs = torch.arange(self.cfg.scene.num_envs)
#         indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
#         self.visualization_markers.visualize(loc, rots, marker_indices=indices)

#     def _pre_physics_step(self, actions: torch.Tensor) -> None:
#         self.actions = actions.clone()
#         self._visualize_markers()

#     def _apply_action(self) -> None:
#         self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

#     def _get_observations(self) -> dict:
#         self.velocity = self.robot.data.root_com_vel_w
#         self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)

#         dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
#         cross = torch.cross(self.forwards, self.commands, dim=-1)[:,-1].reshape(-1,1)
#         forward_speed = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
#         obs = torch.hstack((dot, cross, forward_speed))

#         observations = {"policy": obs}
#         return observations

#     def _get_rewards(self) -> torch.Tensor:
#         forward_reward = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
#         allignment_reward = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
#         total_reward = forward_reward * torch.exp(allignment_reward)
#         return total_reward

#     def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
#         time_out = self.episode_length_buf >= self.max_episode_length - 1

#         return False, time_out

#     def _reset_idx(self, env_ids: Sequence[int] | None):
#         if env_ids is None:
#             env_ids = self.robot._ALL_INDICES
#         super()._reset_idx(env_ids)

#         # pick new commands for reset envs
#         self.commands[env_ids] = torch.randn((len(env_ids), 3)).cuda()
#         self.commands[env_ids,-1] = 0.0
#         self.commands[env_ids] = self.commands[env_ids]/torch.linalg.norm(self.commands[env_ids], dim=1, keepdim=True)

#         # recalculate the orientations for the command markers with the new commands
#         ratio = self.commands[env_ids][:,1]/(self.commands[env_ids][:,0]+1E-8)
#         gzero = torch.where(self.commands[env_ids] > 0, True, False)
#         lzero = torch.where(self.commands[env_ids]< 0, True, False)
#         plus = lzero[:,0]*gzero[:,1]
#         minus = lzero[:,0]*lzero[:,1]
#         offsets = torch.pi*plus - torch.pi*minus
#         self.yaws[env_ids] = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

#         # set the root state for the reset envs
#         default_root_state = self.robot.data.default_root_state[env_ids]
#         default_root_state[:, :3] += self.scene.env_origins[env_ids]

#         self.robot.write_root_state_to_sim(default_root_state, env_ids)
#         self._visualize_markers()
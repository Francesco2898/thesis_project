# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from pxr import UsdGeom

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

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.actions = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)   

        self.success_steps_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_targ_dist = torch.full((self.num_envs,), float("inf"), device=self.device)
                # SummaryWriter per TensorBoard
        self.writer = SummaryWriter(log_dir="runs/my_run")
        self.global_step = 0  # contatore manuale degli step

        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        # auxiliary variables
        self.effort_limits = self.robot.data.joint_effort_limits
        self.robot_dof_lower_limit = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device = self.device)
        self.robot_dof_upper_limit = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device = self.device)
        self.robot_effort_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        self.gripper_fwd_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.gripper_up_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        self.gripper_close_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        self.object_up_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.object_inward_axis = torch.tensor([-1.0, 0.0, 0.0], device=self.device)

        self.gripper_state = torch.ones((self.num_envs), device=self.device) # 1 = open, 0 = closed
        self.robot_local_grasp_pos = (self.gripper_fwd_axis * 0.04).expand(self.num_envs, 3).clone()
        self.robot_local_grasp_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 4).clone()
        self.object_local_grasp_pos = torch.tensor([0.0, 0.0, 0.02], device=self.device).expand(self.num_envs, 3).clone()
        self.object_local_grasp_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 4).clone()

        # robot link idxs
        self.hand_link_idx = self.robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self.robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self.robot.find_bodies("panda_rightfinger")[0][0]
        self.left_finger_joint_idx = self.robot.find_joints("panda_finger_joint1")[0][0]
        self.right_finger_joint_idx = self.robot.find_joints("panda_finger_joint2")[0][0]
        self.finger_joints_ids = torch.tensor(
            [self.left_finger_joint_idx, self.right_finger_joint_idx],
            device=self.device, dtype=torch.long
        )

        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limit)
        self.robot_dof_speed_scales[self.robot.find_joints("panda_finger_joint1")[0]] = 0.3
        self.robot_dof_speed_scales[self.robot.find_joints("panda_finger_joint2")[0]] = 0.3

        # target auxiliary variables
        # sample of the new target in a box space
        self.box_min = torch.tensor([-0.5, -0.5, 0.021], device=self.device)
        self.box_max = torch.tensor([0.5, 0.5, 0.021], device=self.device)
        self.object_init_z = torch.zeros(self.num_envs, device=self.device)
        self.cube_half = 0.02
        self.pad = 0.004

        # place position
        self.place_pos_local = torch.zeros((self.num_envs, 3), device=self.device)

    def _setup_scene(self):

        self.robot = Articulation(self.cfg.robot_cfg)
        print("[DEBUG] Actuator config:")
        print(self.robot.cfg.actuators)
        # target position in local coordinates
        self.target_pos = torch.tensor([[0.5, 0.5, 0.021]], device=self.device).repeat(self.num_envs, 1)
        self.target_rot = torch.zeros((self.num_envs, 4), device=self.device)

        target_obj_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/TargetObject",
            spawn=sim_utils.CuboidCfg(
                size=(0.04, 0.04, 0.04),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 1.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.5, 0.5, 0.021),
                rot=(1.0, 0.0, 0.0, 0.0)
            ),
            debug_vis=True
        )
        self.target_obj = RigidObject(target_obj_cfg)

        # adding a marker following the target
        # target_marker_cfg = VisualizationMarkersCfg(
        #     prim_path="Visuals/target_marker",
        #     markers={
        #         "sphere": sim_utils.SphereCfg(
        #             radius=0.05,
        #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
        #         )
        #     }
        # )
        # self.target_marker = VisualizationMarkers(target_marker_cfg)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["target"] = self.target_obj
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

        arm_actions = self.actions[..., :-2]
        gripper_cmd = self.actions[..., -2:].mean(dim=-1, keepdim=True)
        gripper_cmd = gripper_cmd.expand(-1, 2)
        # arm section action
        targets_arm = self.robot_dof_targets[..., :-2] + self.robot_dof_speed_scales[:-2] * self.dt * arm_actions * self.cfg.action_scale
        self.robot_dof_targets[..., :-2] = torch.clamp(targets_arm, self.robot_dof_lower_limit[:-2], self.robot_dof_upper_limit[:-2])
        # fingers section action
        targets_gripper = self.robot_dof_targets[..., self.finger_joints_ids] + self.robot_dof_speed_scales[self.finger_joints_ids] * self.dt * gripper_cmd * self.cfg.action_scale
        self.robot_dof_targets[..., self.finger_joints_ids] = torch.clamp(targets_gripper, self.robot_dof_lower_limit[self.finger_joints_ids], self.robot_dof_upper_limit[self.finger_joints_ids])

    def _apply_action(self) -> None:

        # Torque Control
        # self.robot.set_joint_effort_target(self.robot_effort_targets, joint_ids=self.dof_idx)

        # Position control
        self.robot.set_joint_position_target(self.robot_dof_targets)


    def _get_observations(self) -> dict:
        # end effector state
        # lf_pos_w = self.robot.data.body_pos_w[:, self.left_finger_link_idx] # position of the left finger in world coord
        # rf_pos_w = self.robot.data.body_pos_w[:, self.right_finger_link_idx] # position of the right finger in world coord
        # lf_vel_w = self.robot.data.body_lin_vel_w[:, self.left_finger_link_idx]
        # rf_vel_w = self.robot.data.body_lin_vel_w[:, self.right_finger_link_idx]
        # grasp_pos_w = (lf_pos_w + rf_pos_w)/2.0 # position of the grasp point in the middle
        # grasp_vel_w = (lf_vel_w + rf_vel_w)/2.0
        # grasp_pos_local = self.get_env_local_pos(self.scene.env_origins, grasp_pos_w)
        # grasp_quat_w = self.robot.data.body_quat_w[:, self.hand_link_idx]
        # ee_ang = self.robot.data.body_ang_vel_w[:, self.hand_link_idx] # angular velocity
        self.compute_intermediate_values()
        robot_grasp_pos_w = self.robot_grasp_pos
        object_grasp_pos_w = self.object_grasp_pos_w

        # to target_vector
        to_target = object_grasp_pos_w - robot_grasp_pos_w

        # orientation error
        q_hand = self.robot_grasp_rot
        # print("[DBG] q_hand shape:", tuple(q_hand.shape), " last_dim:", q_hand.shape[-1])
        q_hand = q_hand/(q_hand.norm(dim=-1, keepdim=True) + 1e-8)
        q_targ = self.object_grasp_rot_w
        q_targ = q_targ/(q_targ.norm(dim=-1, keepdim=True) + 1e-8)

        q_rel = self.quat_mul(q_targ, self.quat_conjugate(q_hand))
        q_rel = torch.where(q_rel[..., :1] < 0, -q_rel, q_rel)

        qv = q_rel[..., 1:4]
        qw = q_rel[..., :1].clamp(min=1e-8)
        nv = torch.norm(qv, dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(nv, qw)
        ori_err = torch.where(nv > 1e-6, angle * (qv / nv), 2.0 * qv)
        ori_err = torch.clamp(ori_err/math.pi, -1.0, 1.0)

        # fingers aperture
        q_f     = self.robot.data.joint_pos[:, self.finger_joints_ids]     # (N,2)
        q_hi = self.robot_dof_upper_limit[self.finger_joints_ids]
        q_lw = self.robot_dof_lower_limit[self.finger_joints_ids]
        q_norm = (q_f - q_lw)/(q_hi - q_lw + 1e-8)
        aperture = q_norm.mean(dim=-1, keepdim=True)
        aperture_centered = 2.0 * aperture - 1.0

        # joint state
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        joint_pos_norm = 2.0 * (joint_pos - self.robot_dof_lower_limit)/(self.robot_dof_upper_limit - self.robot_dof_lower_limit) - 1.0
        joint_vel_norm = joint_vel * self.cfg.dof_velocity_scale

        # --- quaternion hygiene: normalizza + canonalizza emisfero ---
        # ee_quat_w = ee_quat_w / (ee_quat_w.norm(dim=-1, keepdim=True) + 1e-8)
        # tgt_quat_w = tgt_quat_w / (tgt_quat_w.norm(dim=-1, keepdim=True) + 1e-8)
        # same_hemi = ((ee_quat_w * tgt_quat_w).sum(dim=-1, keepdim=True) >= 0)
        # ee_quat_w = torch.where(same_hemi, ee_quat_w, -ee_quat_w)  # riduce discontinuità di segno
        # print("[DEBUG] target dist = ", targ_dist.mean().item())
        obs = torch.hstack((joint_pos_norm,
                            joint_vel_norm,
                            to_target,
                            ori_err,
                            aperture_centered
                            ))
        obs = torch.nan_to_num(obs, nan=0, posinf=1e6, neginf=-1e6)
        observations = {"policy": torch.clamp(obs, -5.0, 5.0)}
        return observations 

    def _get_rewards(self) -> torch.Tensor:
        self.compute_intermediate_values()
        # end effector position in world frame
        # end effector state
        lf_pos_w = self.robot.data.body_pos_w[:, self.left_finger_link_idx] # position of the left finger in world coord
        rf_pos_w = self.robot.data.body_pos_w[:, self.right_finger_link_idx] # position of the right finger in world coord
        lf_vel_w = self.robot.data.body_lin_vel_w[:, self.left_finger_link_idx]
        rf_vel_w = self.robot.data.body_lin_vel_w[:, self.right_finger_link_idx]
        grasp_pos_w = (lf_pos_w + rf_pos_w)/2.0 # position of the grasp point in the middle
        grasp_vel_w = (lf_vel_w + rf_vel_w)/2.0
        grasp_pos_local = self.get_env_local_pos(self.scene.env_origins, grasp_pos_w)
        grasp_quat_w = self.robot.data.body_quat_w[:, self.hand_link_idx]
        ee_ang = self.robot.data.body_ang_vel_w[:, self.hand_link_idx] # angular velocity
        # target
        targ_dist = torch.norm(self.object_grasp_pos_w - self.robot_grasp_pos, p=2, dim=-1)
        targ_quat_w = self.target_obj.data.root_quat_w

        # quaternions normaliozation
        grasp_quat_w = grasp_quat_w / (grasp_quat_w.norm(dim=-1, keepdim=True) + 1e-8)
        targ_quat_w = targ_quat_w / (targ_quat_w.norm(dim=-1, keepdim=True) + 1e-8)
        # close = targ_dist < 0.05
        # next_buf = torch.where(close, self.success_steps_buf+1, torch.zeros_like(self.success_steps_buf))
        # just_succeeded = (next_buf == 5)
        # reward_success = just_succeeded.float() * 10.0
        total_reward = self.compute_rewards(self.actions, targ_dist, lf_pos_w, rf_pos_w)
        self.global_step += 1  # aggiorna il contatore step
        # if self.global_step % 100 == 0:
        #     # print("[DEBUG] Logging step:", self.global_step)
        #     self.writer.add_scalar("Metrics/reward_mean", total_reward.mean().item(), self.global_step)
        #     self.writer.add_scalar("Metrics/target_dist_mean", targ_dist.mean().item(), self.global_step)
        #     self.writer.add_scalar("Metrics/target_dist_min", targ_dist.min().item(), self.global_step)
        #     self.writer.add_scalar("Metrics/target_dist_max", targ_dist.max().item(), self.global_step)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        success_lift = 0.10
        d = self.target_obj.data.root_pos_w[:, 2] - self.object_init_z
        terminated = d > success_lift
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

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
        # self.target_rot[env_ids] = self.random_quat_uniform(len(env_ids))

        new_target_pos_w = new_target_pos_local + self.scene.env_origins[env_ids]
        new_object_rot_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        targ_state = torch.cat((
            new_target_pos_w,
            new_object_rot_w,
            torch.zeros((len(env_ids),  3), device=self.device), # null linear velocity
            torch.zeros((len(env_ids),  3), device=self.device) # null angular velocity
        ), dim=-1)
        # print("[DEBUG] env_ids:", env_ids)
        # print("[DEBUG] targ_state shape:", targ_state.shape)
        # print("[DEBUG] targ_state:", targ_state)
        self.target_obj.write_root_state_to_sim(targ_state, env_ids)
        self.object_init_z[env_ids] = self.target_obj.data.root_pos_w[env_ids, 2]
        self.compute_intermediate_values()
        prev_dist_0 = torch.norm(self.object_grasp_pos_w[env_ids] - self.robot_grasp_pos[env_ids])
        self.prev_targ_dist[env_ids] = prev_dist_0
        # self.target_marker.visua lize((self.target_pos + self.scene.env_origins).cpu())
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
        
    def compute_rewards(self, actions,
                        targ_dist: torch.Tensor,
                        robot_lf_pos: torch.Tensor,
                        robot_rf_pos: torch.Tensor,
                        ):
        # reward based on the distance from the target
        reward_dist = 1.0 / (1.0 + targ_dist**2)
        reward_dist *= reward_dist
        reward_dist = torch.where(targ_dist < 0.02, reward_dist*2, reward_dist)

        # distance improvement
        # if not hasattr(self, "prev_targ_dist"):
        #     self.prev_targ_dist = targ_dist.clone()
        # improvement = torch.clamp(self.prev_targ_dist - targ_dist, min=0.0)
        # reward_progress = 0.8 * improvement
        # self.prev_targ_dist = targ_dist.detach()

        # gating coefficient near the target
        w_near = torch.clamp(1.0 - targ_dist/0.2, 0.0, 1.0)
        # w_near = torch.where(targ_dist < 0.2, 1.0, 0.0)

        # orientation reward
        fwd_axis = self.gripper_fwd_axis.expand(self.num_envs, -1)
        gripper_up_axis = self.gripper_up_axis.expand(self.num_envs, -1)
        object_up_axis = self.object_up_axis.expand(self.num_envs, -1)
        object_inward_axis = self.object_inward_axis.expand(self.num_envs, -1)

        axis1 = self.rotate_vec_by_quat(fwd_axis, self.robot_grasp_rot) # forward gripper (world)
        axis2 = self.rotate_vec_by_quat(gripper_up_axis, self.robot_grasp_rot) # gripper up axis (world)
        axis3 = self.rotate_vec_by_quat(object_up_axis, self.object_grasp_rot_w) # up direction object (world)
        axis4 = self.rotate_vec_by_quat(object_inward_axis, self.object_grasp_rot_w) # inward direction object (world)

        dot1 = (axis1 * (-axis3)).sum(dim=-1).clamp(-1.0, 1.0)
        dot2 = (axis2 * axis4).sum(dim=-1).clamp(-1.0, 1.0)

        rot_term = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)
        reward_rot = w_near * self.cfg.rot_reward_scale * rot_term

        # lift reward
        z = self.target_obj.data.root_pos_w[:, 2]
        lift = torch.clamp(z - self.object_init_z, min=0.0)
        reward_lift = lift * self.cfg.lift_reward_scale

        # finger grasp reward
        gripper_close_axis = self.gripper_close_axis.expand(self.num_envs, -1)
        close_axis_w = self.rotate_vec_by_quat(gripper_close_axis, self.robot_grasp_rot)

        dL = ((robot_lf_pos - self.object_grasp_pos_w) * close_axis_w).sum(dim=-1)
        dR = ((robot_rf_pos - self.object_grasp_pos_w) * close_axis_w).sum(dim=-1)
        d_tar = self.cube_half + self.pad
        errL = torch.abs(dL + d_tar)
        errR = torch.abs(dR - d_tar)

        band = 0.02
        pos_reward = torch.clamp((1.0 - (errL + errR)/(2*band)), min=0.0, max=1.0)
        sym_reward = torch.exp(-0.5 * ((dL + dR) / 0.01)**2) # gaussian reward
        gap_proj = ((robot_rf_pos - robot_lf_pos) * close_axis_w).sum(dim=-1).abs()
        gap_err = torch.abs(gap_proj - 2*d_tar)
        gap_reward = torch.clamp((1.0 - gap_err/(2*self.pad)), min=0.0, max=1.0)

        q_f = self.robot.data.joint_pos[:, self.finger_joints_ids]
        q_lo = self.robot_dof_lower_limit[self.finger_joints_ids]
        q_hi = self.robot_dof_upper_limit[self.finger_joints_ids]
        q_norm = (q_f - q_lo) / (q_hi - q_lo + 1e-8) 
        closed_frac = 1.0 - q_norm.mean(dim=-1)
        near_contact = ((errL < 2*self.pad) & (errR < 2*self.pad)).float()
        close_hint = near_contact * closed_frac

        reward_finger = self.cfg.finger_reward_scale * w_near * (
            0.5 * pos_reward + 0.25 * gap_reward + 0.25 * sym_reward + 0.6 * close_hint
            )
        
        wrong_side = ((dL > 0.0) | (dR < 0.0)).float()
        reward_finger = reward_finger - w_near * self.cfg.finger_reward_scale * 0.2 * wrong_side

        # lfinger_dist = robot_lf_pos[:, 1] - self.object_grasp_pos_w[:, 1]
        # rfinger_dist = self.object_grasp_pos_w[:, 1] - robot_rf_pos[:, 1]
        # finger_dist_penalty = torch.zeros_like(lfinger_dist)
        # finger_dist_penalty += torch.where(lfinger_dist < 0, lfinger_dist, torch.zeros_like(lfinger_dist))
        # finger_dist_penalty += torch.where(rfinger_dist < 0, rfinger_dist, torch.zeros_like(rfinger_dist))
        # reward_finger = w_near * self.cfg.finger_reward_scale * finger_dist_penalty


        # actions pensalty
        penalty_action = torch.sum(actions**2, dim=-1)
        penalty_action = self.cfg.action_penalty_scale * penalty_action

        # compute reward
        reward = (reward_dist
                  - penalty_action 
                  + reward_rot 
                  + reward_lift
                  + reward_finger)

        #reward for lifting the object
        reward = torch.where(lift > 0.01, reward + 0.25, reward)
        reward = torch.where(lift > 0.05, reward + 0.25, reward)
        reward = torch.where(lift > 0.1 , reward + 0.25, reward)

        if self.global_step % 100 == 0:
            self.writer.add_scalar("geom/to_target", targ_dist.mean().item(), self.global_step)
            self.writer.add_scalar("reward/total", reward.mean().item(), self.global_step)
            self.writer.add_scalar("reward/dist", reward_dist.mean().item(), self.global_step)
            self.writer.add_scalar("reward/rot", reward_rot.mean().item(), self.global_step)
            self.writer.add_scalar("reward/lift", reward_lift.mean().item(), self.global_step)
            self.writer.add_scalar("reward/fingers", reward_finger.mean().item(), self.global_step)
    
        return reward
    
    def random_quat_uniform(self, n: int) -> torch.Tensor:
        u1 = torch.rand(n, device=self.device)
        u2 = torch.rand(n, device=self.device)
        u3 = torch.rand(n, device=self.device)

        sqrt1_u1 = torch.sqrt(1.0 - u1)
        sqrt_u1  = torch.sqrt(u1)
    
        qx = sqrt1_u1 * torch.sin(2.0 * math.pi * u2)
        qy = sqrt1_u1 * torch.cos(2.0 * math.pi * u2)
        qz = sqrt_u1  * torch.sin(2.0 * math.pi * u3)
        qw = sqrt_u1  * torch.cos(2.0 * math.pi * u3)
    
        q = torch.stack((qw, qx, qy, qz), dim=-1)
        return q
    
    def rotate_vec_by_quat(self, v: torch.Tensor, q: torch.Tensor):  # v: (...,3), q: (...,4) [w,x,y,z]
        w, x, y, z = q.unbind(-1)
        qv = torch.stack((x, y, z), dim=-1)
        t  = 2.0 * torch.cross(qv, v, dim=-1)
        return v + w.unsqueeze(-1)*t + torch.cross(qv, t, dim=-1)
    
    def compute_intermediate_values(self, env_ids : torch.Tensor | None = None ):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # hand pose (world)
        hand_pos_w = self.robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_quat_w = self.robot.data.body_quat_w[env_ids, self.hand_link_idx]

        # object pose (world)
        obj_pos_w = self.target_obj.data.root_pos_w[env_ids]
        obj_quat_w = self.target_obj.data.root_quat_w[env_ids]

        # robot grasp world = hand_pos + R(hand)*local_offset
        rg_local = self.robot_local_grasp_pos[env_ids]
        rg_rot_local = self.robot_local_grasp_rot[env_ids]
        rg_offset_world = self.rotate_vec_by_quat(rg_local, hand_quat_w)
        self.robot_grasp_pos[env_ids] = hand_pos_w + rg_offset_world
        self.robot_grasp_rot[env_ids] = hand_quat_w

        og_local = self.object_local_grasp_pos[env_ids]
        og_offset_world = self.rotate_vec_by_quat(og_local, obj_quat_w)
        self.object_grasp_pos_w = torch.zeros((self.num_envs, 3), device=self.device) if not hasattr(self, "object_grasp_pos_w") else self.object_grasp_pos_w
        self.object_grasp_rot_w = torch.zeros((self.num_envs, 4), device=self.device) if not hasattr(self, "object_grasp_rot_w") else self.object_grasp_rot_w
        self.object_grasp_pos_w[env_ids] = obj_pos_w + og_offset_world
        self.object_grasp_rot_w[env_ids] = obj_quat_w

    def quat_conjugate(self, q: torch.Tensor):
        w, x, y, z = q.unbind(-1)
        q_conj = torch.stack((w, -x, -y, -z), dim=-1)
        return q_conj
    
    def quat_mul(self, q1: torch.Tensor, q2: torch.Tensor):  # ([...,4] * [...,4]) -> [...,4], conv. [w,x,y,z]
        w1,x1,y1,z1 = q1.unbind(-1)
        w2,x2,y2,z2 = q2.unbind(-1)
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return torch.stack((w,x,y,z), dim=-1)
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
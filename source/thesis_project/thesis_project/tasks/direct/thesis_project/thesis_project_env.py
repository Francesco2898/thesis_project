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

        self.ee_offset = 0.03
        self.gripper_fwd_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.gripper_up_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        self.gripper_close_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        self.object_up_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.object_inward_axis = torch.tensor([-1.0, 0.0, 0.0], device=self.device)

        self.cube_half = 0.02
        self.gripper_state = torch.ones((self.num_envs), device=self.device) # 1 = open, 0 = closed
        self.gap_finger = torch.zeros((self.num_envs, 1), device=self.device)
        # self.robot_local_grasp_pos = (self.gripper_fwd_axis * 0.1).expand(self.num_envs, 3).clone()
        self.robot_local_grasp_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.robot_local_grasp_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 4).clone()
        self.object_local_grasp_pos = torch.tensor([0.0, 0.0, self.cube_half], device=self.device).expand(self.num_envs, 3).clone()
        self.object_local_grasp_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 4).clone()
        self.object_grasp_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_grasp_rot_w = torch.zeros((self.num_envs, 4), device=self.device)

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
        self.robot_dof_speed_scales[self.robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self.robot.find_joints("panda_finger_joint2")[0]] = 0.1

        # target auxiliary variables
        # sample of the new target in a box space
        # self.box_min = torch.tensor([-0.4, -0.4, 0.021], device=self.device)
        # self.box_max = torch.tensor([0.4, 0.4, 0.021], device=self.device)
        self.box_min = torch.tensor([0.2, -0.3, 0.021], device=self.device)
        self.box_max = torch.tensor([0.6,  0.3, 0.021], device=self.device)
        self.object_init_z = torch.zeros(self.num_envs, device=self.device)
        self.r_in, self.r_out = 0.025, 0.05
        self.near_gate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ori_th = 0.75
        self.ori_gate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success_height = 0.005

        self.has_object = torch.zeros(self.num_envs,dtype=torch.bool, device=self.device)
        self.prev_has_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.place_ori = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 4)

    def _setup_scene(self):

        self.robot = Articulation(self.cfg.robot_cfg)
        print("[DEBUG] Actuator config:")
        print(self.robot.cfg.actuators)
        # target position in local coordinates
        self.target_pos = torch.tensor([[0.5, 0.5, 0.021]], device=self.device).repeat(self.num_envs, 1)

        self.place_pos_local = torch.tensor([[-0.5, -0.5, 0.021]], device=self.device).repeat(self.num_envs, 1)

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

        # adding a marker following the end_effector
        target_marker_cfg = VisualizationMarkersCfg(
            prim_path="Visuals/target_marker",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
                )
            }
        )
        self.ee_marker = VisualizationMarkers(target_marker_cfg)
        self.object_marker = VisualizationMarkers(target_marker_cfg)
        self.place_pos_marker = VisualizationMarkers(target_marker_cfg)
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
        # arm section action
        targets_arm = self.robot_dof_targets[..., :-2] + self.robot_dof_speed_scales[:-2] * self.dt * arm_actions * self.cfg.action_scale
        self.robot_dof_targets[..., :-2] = torch.clamp(targets_arm, self.robot_dof_lower_limit[:-2], self.robot_dof_upper_limit[:-2])

        # gripper_cmd = self.actions[..., -2:].mean(dim=-1, keepdim=True)
        gripper_cmd = self.actions[..., -2:-1].clamp(-1.0, 1.0)
        gripper_cmd = gripper_cmd.expand(-1, 2)
        OPEN = 1.0
        CLOSE = -1.0
        # grasp_phase_ok = (self.near_gate & self.ori_gate)

        dist_pick = torch.norm(self.object_grasp_pos_w - self.robot_grasp_pos, dim=-1)
        dist_place = torch.norm(self.place_pos_w - self.target_obj.data.root_pos_w, dim=-1)
        pick_cnd = dist_pick < self.r_in
        place_cnd = dist_place < self.r_in
        pick_task = ~self.has_object
        place_task = self.has_object
        close_gripper = pick_task & pick_cnd
        open_gripper  = (place_task & place_cnd) | (pick_task & (dist_pick > self.r_in))

        self.gripper_state = torch.where(close_gripper, torch.zeros_like(self.gripper_state), self.gripper_state)
        self.gripper_state = torch.where(open_gripper, torch.ones_like(self.gripper_state), self.gripper_state)
        cmd = torch.where(self.gripper_state.unsqueeze(-1) > 0.5, torch.full_like(gripper_cmd, OPEN), torch.full_like(gripper_cmd, CLOSE))
        # fingers section action
        targets_gripper = self.robot_dof_targets[..., self.finger_joints_ids] + self.robot_dof_speed_scales[self.finger_joints_ids] * self.dt * cmd * self.cfg.action_scale
        self.robot_dof_targets[..., self.finger_joints_ids] = torch.clamp(targets_gripper, self.robot_dof_lower_limit[self.finger_joints_ids], self.robot_dof_upper_limit[self.finger_joints_ids])

    def _apply_action(self) -> None:

        # Torque Control
        # self.robot.set_joint_effort_target(self.robot_effort_targets, joint_ids=self.dof_idx)

        # Position control
        self.robot.set_joint_position_target(self.robot_dof_targets)


    def _get_observations(self) -> dict:

        self.compute_intermediate_values()

        phase_mask = self.has_object.unsqueeze(-1)
        robot_grasp_pos_w = self.robot_grasp_pos
        goal_pos_w = torch.where(phase_mask, self.place_pos_w, self.object_grasp_pos_w)
        # to target_vector
        to_target = goal_pos_w - robot_grasp_pos_w

        # orientation error
        q_hand = self.robot_grasp_rot
        # print("[DBG] q_hand shape:", tuple(q_hand.shape), " last_dim:", q_hand.shape[-1])
        q_hand = q_hand/(q_hand.norm(dim=-1, keepdim=True) + 1e-8)
        q_targ = torch.where(phase_mask, self.place_ori, self.object_grasp_rot_w)
        q_targ = q_targ/(q_targ.norm(dim=-1, keepdim=True) + 1e-8)

        q_rel = self.quat_mul(q_targ, self.quat_conjugate(q_hand))
        q_rel = torch.where(q_rel[..., :1] < 0, -q_rel, q_rel)

        qv = q_rel[..., 1:4]
        qw = q_rel[..., :1].clamp(min=1e-8)
        nv = torch.norm(qv, dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(nv, qw)
        ori_err = torch.where(nv > 1e-6, angle * (qv / nv), 2.0 * qv)
        ori_err = torch.clamp(ori_err/math.pi, -1.0, 1.0)

        # lift state
        z = self.target_obj.data.root_pos_w[:, 2]
        lift = torch.clamp((z - self.object_init_z), min=0.0, max=self.success_height)
        lift_height = (lift/self.success_height).unsqueeze(-1)

        # joint state
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        joint_pos_norm = 2.0 * (joint_pos - self.robot_dof_lower_limit)/(self.robot_dof_upper_limit - self.robot_dof_lower_limit) - 1.0
        joint_vel_norm = joint_vel * self.cfg.dof_velocity_scale

        obs = torch.hstack((joint_pos_norm,
                            joint_vel_norm,
                            phase_mask,
                            to_target,
                            ori_err,
                            lift_height
                            ))
        obs = torch.nan_to_num(obs, nan=0, posinf=1e6, neginf=-1e6)
        observations = {"policy": torch.clamp(obs, -5.0, 5.0)}
        return observations 

    def _get_rewards(self) -> torch.Tensor:
        self.compute_intermediate_values()
        # to goal
        to_goal = self.place_pos_w - self.target_obj.data.root_pos_w
        # target
        to_target = self.object_grasp_pos_w - self.robot_grasp_pos

        total_reward = self.compute_rewards(self.actions, to_goal, to_target)
        self.global_step += 1  # update the step counter
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        goal_dist = torch.norm(self.place_pos_w - self.target_obj.data.root_pos_w, dim=-1)
        goal_err_dist = 0.02
        terminated = (goal_dist < goal_err_dist) & (~self.has_object) 
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # robot reset
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

        # target reset
        # local position wrt the env
        new_target_pos_local = sample_uniform(self.box_min, self.box_max, (len(env_ids), 3), device=self.device)
        self.target_pos[env_ids] = new_target_pos_local

        new_target_pos_w = new_target_pos_local + self.scene.env_origins[env_ids]
        new_object_rot_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        targ_state = torch.cat((
            new_target_pos_w,
            new_object_rot_w,
            torch.zeros((len(env_ids),  3), device=self.device), # null linear velocity
            torch.zeros((len(env_ids),  3), device=self.device) # null angular velocity
        ), dim=-1)

        self.target_obj.write_root_state_to_sim(targ_state, env_ids)
        self.object_init_z[env_ids] = self.target_obj.data.root_pos_w[env_ids, 2]

        # goal reset
        new_place_pos_local = sample_uniform(self.box_min, self.box_max, (len(env_ids), 3), device=self.device)
        self.place_pos_local[env_ids] = new_place_pos_local

        self.compute_intermediate_values()

        prev_dist_0 = torch.norm(self.object_grasp_pos_w[env_ids] - self.robot_grasp_pos[env_ids], dim=-1)
        self.prev_targ_dist[env_ids] = prev_dist_0

        self.gripper_state[env_ids] = 1.0
        self.hold_steps[env_ids] = 0
        self.has_object[env_ids] = False
        self.prev_has_object[env_ids] = False

    def get_env_local_pos(self, env_pos: torch.Tensor, world_pos: torch.Tensor, world_quat: torch.Tensor | None = None):
        # compute position in environment local coordinate system
        local_pose = world_pos - env_pos
        if world_quat == None: # interested only in the position
            return local_pose
        else: # interested also in the orientation
            qx, qy, qz, qw = world_quat.unbind(-1)
            return torch.stack([local_pose[:, 0], local_pose[:, 1], local_pose[:, 2], qw, qx, qy, qz],dim=-1)
        
    def compute_rewards(self, actions,
                        to_goal: torch.Tensor,
                        to_target: torch.Tensor
                        ):
        # reward based on the distance ee <-> object
        reward_reach, targ_dist = self.reward_dist(to_target)
        # reward based on the distance object <-> goal_position
        reward_place, goal_dist = self.reward_dist(to_goal)

        # distance improvement
        improvement = torch.clamp(self.prev_targ_dist - targ_dist, min=0.0)
        reward_progress = torch.where(targ_dist < 0.08, 30.0 * improvement, 0.0)
        self.prev_targ_dist = targ_dist.detach()

        # gating coefficient near the target
        r_soft  = 0.3  # inizia a contare già a 30 cm
        r_close = 0.12  # “conteggio pieno” sotto 12 cm
        w_rot = torch.clamp((r_soft - targ_dist)/(r_soft - r_close + 1e-8), 0.0, 1.0) # w_rot -> starts to allign relatively close to the target
        # w_rot = torch.clamp(1.0 - targ_dist/0.25, 0.0, 1.0)
        enter = (targ_dist < self.r_in)
        out = (targ_dist > self.r_out)
        near_prev = self.near_gate # previous step condition
        zeros = torch.zeros_like(near_prev, dtype=torch.bool)
        ones = torch.ones_like(near_prev, dtype=torch.bool)
        near = torch.where(enter, ones, torch.where(out, zeros, near_prev)) # new step condition
        self.near_gate = near

        # orientation reward
        fwd_axis           = self.gripper_fwd_axis.expand(self.num_envs, -1)
        gripper_up_axis    = self.gripper_up_axis.expand(self.num_envs, -1)
        object_up_axis     = self.object_up_axis.expand(self.num_envs, -1)
        object_inward_axis = self.object_inward_axis.expand(self.num_envs, -1)

        axis1 = self.rotate_vec_by_quat(fwd_axis, self.robot_grasp_rot) # forward gripper (world)
        axis2 = self.rotate_vec_by_quat(gripper_up_axis, self.robot_grasp_rot) # gripper up axis (world)
        axis3 = self.rotate_vec_by_quat(object_up_axis, self.object_grasp_rot_w) # up direction object (world)
        axis4 = self.rotate_vec_by_quat(object_inward_axis, self.object_grasp_rot_w) # inward direction object (world)

        dot1 = (axis1 * (-axis3)).sum(dim=-1).clamp(-1.0, 1.0)
        dot2 = (axis2 * axis4).sum(dim=-1).clamp(-1.0, 1.0)

        dot1_p = torch.clamp(dot1, 0.0, 1.0)
        dot2_p = torch.clamp(dot2, 0.0, 1.0)

        rot_term = 0.5 * (dot1_p**2 + dot2_p**2)
        # rot_term = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)
        reward_rot = w_rot * self.cfg.rot_reward_scale * rot_term
        # gating on the orientation
        # in_ori_th = rot_term > self.ori_th
        # ori_ok = in_ori_th
        # self.ori_gate = ori_ok
        
        # lift reward
        z = self.target_obj.data.root_pos_w[:, 2]
        lift = torch.clamp(z - self.object_init_z, min=0.0)  # quanto si è alzato il cubo
        lift_norm = torch.clamp(lift / self.success_height, 0.0, 1.0)

        lifted = lift >= 0.001
        hold_cond = (lifted & (targ_dist < self.r_in))
        self.hold_steps = torch.where(hold_cond, self.hold_steps + 1, torch.zeros_like(self.hold_steps))
        HOLD_MIN_STEP = 5
        self.has_object = self.hold_steps >= HOLD_MIN_STEP

        phase = self.has_object.float()
        reward_reach = (1.0 - phase)*reward_reach
        reward_place = phase * reward_place * self.cfg.goal_reward_scale
        reward_progress = (1.0 - phase) * reward_progress
        reward_lift = self.cfg.lift_reward_scale * lift_norm

        # grasp reward
        grasp_success = ((~self.prev_has_object) & (self.has_object))
        # reward_grasp = self.cfg.grasp_reward_scale * grasp_success.float()
        self.prev_has_object = self.has_object.clone()

        # actions pensalty
        penalty_action = torch.sum(actions**2, dim=-1)
        penalty_action = self.cfg.action_penalty_scale * penalty_action

        # compute reward
        reward = (reward_reach
                  + reward_place
                  + reward_progress
                  + reward_rot
                  + reward_lift
                  - penalty_action )

        if self.global_step % 100 == 0:
            self.writer.add_scalar("geom/reach_dist_mean", targ_dist.mean().item(), self.global_step)
            self.writer.add_scalar("geom/goal_dist_mean",  goal_dist.mean().item(),  self.global_step)
            self.writer.add_scalar("geom/rot_term", rot_term.mean().item(), self.global_step)
            self.writer.add_scalar("geom/lift", lift.mean().item(), self.global_step)
            self.writer.add_scalar("geom/has_object_mean", phase.mean().item(), self.global_step)

            self.writer.add_scalar("reward/total", reward.mean().item(), self.global_step)
            self.writer.add_scalar("reward/reward_reach", reward_reach.mean().item(), self.global_step)
            self.writer.add_scalar("reward/reward_place", reward_place.mean().item(), self.global_step)
            self.writer.add_scalar("reward/progress", reward_progress.mean().item(), self.global_step)
            self.writer.add_scalar("reward/rot", reward_rot.mean().item(), self.global_step)
            self.writer.add_scalar("reward/lift", reward_lift.mean().item(), self.global_step)
            # self.writer.add_scalar("reward/grasp", reward_grasp.mean().item(), self.global_step)
    
        return reward
    
    def compute_intermediate_values(self, env_ids : torch.Tensor | None = None ):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # hand pose (world)
        hand_pos_w = self.robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_quat_w = self.robot.data.body_quat_w[env_ids, self.hand_link_idx]
        hand_quat_w = hand_quat_w / (hand_quat_w.norm(dim=-1, keepdim=True) + 1e-8)

        # fingers pos (world)
        lf_pos_w = self.robot.data.body_pos_w[env_ids, self.left_finger_link_idx]
        rf_pos_w = self.robot.data.body_pos_w[env_ids, self.right_finger_link_idx]
        midpoint_w = 0.5 * (lf_pos_w + rf_pos_w)
        midpoint_local = self.rotate_vec_by_quat((midpoint_w - hand_pos_w), self.quat_conjugate(hand_quat_w)) # fingers have the same orientation of the hand
        midpoint_local = midpoint_local + self.ee_offset * self.gripper_fwd_axis
        self.robot_local_grasp_pos[env_ids] = midpoint_local
        gap_finger = (rf_pos_w - lf_pos_w).norm(dim=-1, keepdim=True)
        self.gap_finger[env_ids] = gap_finger

        # object pose (world)
        obj_pos_w = self.target_obj.data.root_pos_w[env_ids]
        obj_quat_w = self.target_obj.data.root_quat_w[env_ids]
        obj_quat_w = obj_quat_w / (obj_quat_w.norm(dim=-1, keepdim=True) + 1e-8)

        # placing position (world)
        self.place_pos_w = self.place_pos_local + self.scene.env_origins

        # robot grasp world = hand_pos + R(hand)*local_offset
        rg_local = self.robot_local_grasp_pos[env_ids]
        # rg_rot_local = self.robot_local_grasp_rot[env_ids]
        rg_offset_world = self.rotate_vec_by_quat(rg_local, hand_quat_w)
        self.robot_grasp_pos[env_ids] = hand_pos_w + rg_offset_world
        self.robot_grasp_rot[env_ids] = hand_quat_w

        og_local = self.object_local_grasp_pos[env_ids]
        og_offset_world = self.rotate_vec_by_quat(og_local, obj_quat_w)
        self.object_grasp_pos_w[env_ids] = obj_pos_w + og_offset_world
        self.object_grasp_rot_w[env_ids] = obj_quat_w

        self.ee_marker.visualize(self.robot_grasp_pos.detach().cpu())
        self.object_marker.visualize(self.object_grasp_pos_w.detach().cpu())
        self.place_pos_marker.visualize(self.place_pos_w.detach().cpu())

    def reward_dist(self, dist_vec: torch.Tensor):
        dist = torch.norm(dist_vec, p=2, dim=-1)
        dx = dist_vec[:, 0]
        dy = dist_vec[:, 1]
        dz = dist_vec[:, 2]

        wx = 10.0  # es. 10.0
        wy = 5.0  # es. 5.0
        wz = 1.0  # es. 1.0
        coord_cost = wx * dx*dx + wy * dy*dy + wz * dz*dz
        reward_dist = (1.0 / (1.0 + coord_cost))
        reward_dist *= reward_dist
        
        # reward_dist = 1.0 / (1.0 + targ_dist**2)
        # reward_dist *= reward_dist
        reward_dist = torch.where(dist < 0.05, reward_dist*2.0, reward_dist)
        return reward_dist, dist

    def quat_conjugate(self, q: torch.Tensor):
        w, x, y, z = q.unbind(-1)
        q_conj = torch.stack((w, -x, -y, -z), dim=-1)
        return q_conj
    
    def quat_mul(self, q1: torch.Tensor, q2: torch.Tensor):
        # q = q1 ⊗ q2 (rotazione di q2 seguita da q1)
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        # formula con v=(x,y,z), w scalare:
        # w = w1*w2 - v1·v2
        w = w1*w2 - (x1*x2 + y1*y2 + z1*z2)
        # v = w1*v2 + w2*v1 + v1×v2
        vx = w1*x2 + w2*x1 + (y1*z2 - z1*y2)
        vy = w1*y2 + w2*y1 + (z1*x2 - x1*z2)
        vz = w1*z2 + w2*z1 + (x1*y2 - y1*x2)
        return torch.stack((w, vx, vy, vz), dim=-1)
    
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


# @torch.jit.script
# def compute_rewards(
#     rew_scale_alive: float,
#     rew_scale_terminated: float,
#     rew_scale_pole_pos: float,
#     rew_scale_cart_vel: float,
#     rew_scale_pole_vel: float,
#     pole_pos: torch.Tensor,
#     pole_vel: torch.Tensor,
#     cart_pos: torch.Tensor,
#     cart_vel: torch.Tensor,
#     reset_terminated: torch.Tensor,
# ):
#     rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
#     rew_termination = rew_scale_terminated * reset_terminated.float()
#     rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
#     rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
#     rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
#     total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
#     return total_reward
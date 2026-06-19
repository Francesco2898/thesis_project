from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import pytorch_kinematics as pk
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import RigidObjectCfg, RigidObject
from torch.utils.tensorboard import SummaryWriter

from .safety_shielding_env_cfg import SafetyEnvCfg

class SafetyEnv(DirectRLEnv):
    cfg: SafetyEnvCfg

    def __init__(self, cfg: SafetyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.global_step = 0
        self.writer = SummaryWriter(log_dir="runs/policy2")

        self.arm_joint_names = self.robot.joint_names[:-2]
        # self.arm_ids, _ = self.robot.find_joints(self.arm_joint_names, preserve_order=True)
        self.arm_ids = torch.arange(self.robot.num_joints - 2, device=self.device, dtype=torch.long)

        self.actions = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.robot_dof_lower_limit = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limit = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.joint_range = torch.clamp(self.robot_dof_upper_limit - self.robot_dof_lower_limit, min=1e-6)
        self.q_goal = sample_uniform(
            self.robot_dof_lower_limit[self.arm_ids],
            self.robot_dof_upper_limit[self.arm_ids],
            (self.num_envs, len(self.arm_ids)),
            device=self.device
        )
        self.prev_q_err = torch.zeros((self.num_envs, len(self.arm_ids)), device=self.device)
        self.tau_max = self.robot.data.joint_effort_limits.to(device=self.device)
        self.tau_cmd = torch.zeros((self.num_envs, len(self.arm_ids)), device=self.device)
        # robot link indices
        self.left_finger_link_idx = self.robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self.robot.find_bodies("panda_rightfinger")[0][0]
        self.hand_link_idx = self.robot.find_bodies("panda_link7")[0][0]
        self.frame_link_2_idx = self.robot.find_bodies("panda_link2")[0][0]
        self.frame_link_4_idx = self.robot.find_bodies("panda_link4")[0][0]
        self.frame_link_6_idx = self.robot.find_bodies("panda_link6")[0][0]
        self.gripper_body_idx = self.robot.find_bodies("panda_hand")[0][0]
        self.frame_link_ids   = torch.tensor(
            [self.frame_link_2_idx, self.frame_link_4_idx, self.frame_link_6_idx, self.hand_link_idx],
            device=self.device, dtype=torch.long
        )
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limit)* 0.5
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_offset = 0.03
        self.gap_finger = torch.zeros((self.num_envs, 1), device=self.device)
        self.gripper_fwd_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        self.probes_w_buf = torch.zeros(self.num_envs, 20, 3, device=self.device)
        self.clear_vec_buf = torch.zeros(self.num_envs, 20, device=self.device)
        self.min_dist_buf = torch.zeros(self.num_envs, device=self.device)

        self.obstacle_box_min = torch.tensor([0.1, -0.35, self.cfg.table_height+self.cfg.obstacle_height/2], device=self.device)
        self.obstacle_box_max = torch.tensor([0.6, 0.35, self.cfg.table_height+self.cfg.obstacle_height/2], device=self.device)
        self.obs_std_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)

        # Obstacle configuration
        def obstacle_cfg( name : str, pos : tuple[float, float, float]):
            return RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/{name}",
                spawn=sim_utils.CylinderCfg(
                    radius=self.cfg.obstacle_radius,
                    height=self.cfg.obstacle_height,
                    axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.6, 1.0)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos= pos,
                    rot= (1.0, 0.0, 0.0, 0.0),
                ),
                debug_vis=True
            )
        probe_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/probe_points",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=self.cfg.link_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 1.0, 0.0)  # giallo
                    ),
                )
            },
        )
        self.obstacle1 = RigidObject(obstacle_cfg("Obstacle1", (0.4, 0.0, self.cfg.table_height + 0.5*self.cfg.obstacle_height)))
        self.obstacle2 = RigidObject(obstacle_cfg("Obstacle2", (0.6, -0.25, self.cfg.table_height + 0.5*self.cfg.obstacle_height)))
        self.obstacle3 = RigidObject(obstacle_cfg("Obstacle3", (0.6, 0.25, self.cfg.table_height + 0.5*self.cfg.obstacle_height)))
        # adding the components to the scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["obstacle1"] = self.obstacle1
        self.scene.rigid_objects["obstacle2"] = self.obstacle2
        self.scene.rigid_objects["obstacle3"] = self.obstacle3
        self.probe_marker = VisualizationMarkers(probe_marker_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self.actions = self.actions[:, self.arm_ids]
        # tau_scale = 0.2
        # self.tau_cmd = self.actions * self.tau_max[0, self.arm_ids] * tau_scale
        # self.tau_cmd = torch.clamp(self.tau_cmd, -self.tau_max[0, self.arm_ids], self.tau_max[0, self.arm_ids])
        targets_arm = self.robot_dof_targets[:, self.arm_ids] + self.robot_dof_speed_scales[self.arm_ids] * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:, self.arm_ids] = torch.clamp(targets_arm, self.robot_dof_lower_limit[self.arm_ids], self.robot_dof_upper_limit[self.arm_ids])
    
    def _apply_action(self) -> None:
        # Torque control
        # self.robot.set_joint_effort_target(self.tau_cmd, joint_ids=self.arm_ids)
        self.robot.set_joint_position_target(self.robot_dof_targets[:, self.arm_ids], joint_ids=self.arm_ids)


    def _get_observations(self):
        joint_pos = self.robot.data.joint_pos[:, self.arm_ids]
        joint_vel = self.robot.data.joint_vel[:, self.arm_ids]
        joint_pos_norm = 2.0 * (joint_pos - self.robot_dof_lower_limit[self.arm_ids])/(self.robot_dof_upper_limit[self.arm_ids] - self.robot_dof_lower_limit[self.arm_ids]) - 1.0
        joint_vel_norm = joint_vel * self.cfg.dof_velocity_scale
        q_err_norm = (self.q_goal - joint_pos)/self.joint_range[self.arm_ids]

        #recomputing probes points and clearances
        self.compute_ee_pos()
        self.probes_w_buf = self.compute_probe_points_pos()
        _, self.clear_vec_buf = self.compute_min_obstacle_distance(self.probes_w_buf)
        scale = 0.2
        clear_obs = torch.tanh(self.clear_vec_buf/scale)

        obs = torch.hstack((joint_pos_norm,
                            joint_vel_norm,
                            q_err_norm,
                            clear_obs
                            ))
        obs = torch.nan_to_num(obs, nan=0, posinf=1e6, neginf=-1e6)
        observations = {"policy": torch.clamp(obs, -5.0, 5.0)}
        return observations
    
    def _get_rewards(self):
        q = self.robot.data.joint_pos[:, self.arm_ids]
        dq = self.robot.data.joint_vel[:, self.arm_ids]

        self.compute_ee_pos()
        self.probes_w_buf = self.compute_probe_points_pos()
        self.min_dist_buf, self.clear_vec_buf = self.compute_min_obstacle_distance(self.probes_w_buf)

        q_err_norm = (self.q_goal - q)/self.joint_range[self.arm_ids]
        # q_dist = torch.norm(q_err_norm, dim=-1)
        q_err = torch.norm(q_err_norm, dim=-1)
        prev_q_err = torch.norm(self.prev_q_err, dim=-1)
        reward_progress = (prev_q_err - q_err) * 2.0

        q_abs = torch.abs(q_err_norm)
        q_worst = torch.max(q_abs, dim=-1).values
        mean_sq = torch.mean(q_abs**2, dim=-1)
        reward_q = torch.exp(-10.0 * mean_sq)
        penalty_q_worst = 1.0 * q_worst
        # reward_for_joint = 1.0/(1.0 + (q_abs)**2)
        # reward_for_joint = torch.where(q_dist < 0.2, 2.0 * reward_for_joint, reward_for_joint)
        # reward_joint = torch.sum(reward_for_joint, dim=-1)*0.1

        # actions penalty
        penalty_action = torch.sum(self.actions**2, dim=-1)
        penalty_action = penalty_action * self.cfg.action_penalty_scale
        # joint_velocity penalty
        near_goal = (q_err <= 0.15).float()
        penalty_dq = (0.0005 + 0.002 * near_goal) * torch.sum(dq**2, dim=-1)
        # obstacle penalty
        d_warn = 0.05
        penalty_obs = torch.relu(d_warn - self.clear_vec_buf)**2
        penalty_obs = torch.sum(penalty_obs, dim=-1) * 50.0

        reward = (reward_q
                  + reward_progress
                  - penalty_dq
                  - penalty_q_worst
                  - penalty_action
                  - penalty_obs)
        if self.global_step % 100 == 0:
            self.writer.add_scalar("geom/q_dist", q_err.mean().item(), self.global_step)
            self.writer.add_scalar("geom/min_clearance_mean", self.min_dist_buf.mean().item(), self.global_step)
            for i in range(len(self.arm_ids)):
                self.writer.add_scalar(f"geom/q_err_norm_joint_{i}", q_err_norm[:, i].abs().mean().item(), self.global_step)
            self.writer.add_scalar("reward/reward_joint", reward_q.mean().item(), self.global_step)
            self.writer.add_scalar("reward/reward_progress", reward_progress.mean().item(), self.global_step)
            self.writer.add_scalar("reward/total", reward.mean().item(), self.global_step)
            self.writer.add_scalar("reward/penalty_dq", penalty_dq.mean().item(), self.global_step)
            self.writer.add_scalar("reward/penalty_action", penalty_action.mean().item(), self.global_step)
            self.writer.add_scalar("reward/penalty_obs", penalty_obs.mean().item(), self.global_step)
            self.writer.add_scalar("reward/penalty_q_worst", penalty_q_worst.mean().item(), self.global_step)
        self.global_step += 1
        self.prev_q_err.copy_(q_err_norm)
        return reward
    
    def _get_dones(self):
        terminated = torch.zeros(self.num_envs,  dtype=bool, device=self.device)
        truncated = self.episode_length_buf >= self.max_episode_length-1
        return terminated, truncated
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        q_goal_pin = torch.tensor(
            [0.2563881650718601, 0.4500463124520452, 0.3697866797164865, -1.9843721466291926, -0.23336613369861603, 2.393948550913406, 1.5501551389393688],
            device=self.device,
            dtype=torch.float32,
        )
        # joint_pos = q_goal_pin.unsqueeze(0).expand(len(env_ids), -1)
        joint_pos = self.robot.data.default_joint_pos[env_ids] + sample_uniform(
            -self.cfg.delta_q,
            self.cfg.delta_q,
            (len(env_ids), self.robot.num_joints),
            device= self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limit, self.robot_dof_upper_limit)
        joint_vel = torch.zeros_like(joint_pos)
        joint_vel[:, self.arm_ids] = sample_uniform(-0.5,
                                   0.5,
                                   (len(env_ids), len(self.arm_ids)),
                                   device=self.device)
        joint_vel[:, -2:] = 0.0
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot_dof_targets[env_ids] = joint_pos
        self.tau_cmd[env_ids] = 0.0
        # self.q_goal[env_ids] = sample_uniform(
        #     self.robot_dof_lower_limit[self.arm_ids],
        #     self.robot_dof_upper_limit[self.arm_ids],
        #     (len(env_ids), len(self.arm_ids)),
        #     device=self.device
        # )
        # self.q_goal[env_ids] = joint_pos[:, self.arm_ids] + sample_uniform(
        #                                         -self.cfg.delta_q,
        #                                          self.cfg.delta_q,    
        #                                          (len(env_ids), len(self.arm_ids)),
        #                                          device=self.device
        #                                          )
        self.q_goal[env_ids] = q_goal_pin.unsqueeze(0).expand(len(env_ids), -1)
        self.q_goal[env_ids] = torch.clamp(self.q_goal[env_ids],
                                           self.robot_dof_lower_limit[self.arm_ids],
                                           self.robot_dof_upper_limit[self.arm_ids]
                                           )
        self.prev_q_err[env_ids] = (self.q_goal[env_ids] - joint_pos[:, self.arm_ids])/self.joint_range[self.arm_ids]
        self.probes_w_buf[env_ids] = 0
        self.clear_vec_buf[env_ids] = 0
        self.min_dist_buf[env_ids] = self.cfg.safe_margin
        self.compute_ee_pos()

        obs1_new_pos = sample_uniform(self.obstacle_box_min, self.obstacle_box_max, (len(env_ids), 3), device=self.device)
        obs1_new_pos = obs1_new_pos + self.scene.env_origins[env_ids]
        obs2_new_pos = sample_uniform(self.obstacle_box_min, self.obstacle_box_max, (len(env_ids), 3), device=self.device)
        obs2_new_pos = obs2_new_pos + self.scene.env_origins[env_ids]
        obs3_new_pos = sample_uniform(self.obstacle_box_min, self.obstacle_box_max, (len(env_ids), 3), device=self.device)
        obs3_new_pos = obs3_new_pos + self.scene.env_origins[env_ids]
        obs1_new_state = torch.cat((
            obs1_new_pos,
            self.obs_std_orientation.repeat(len(env_ids), 1),
            torch.zeros((len(env_ids), 3), device=self.device),
            torch.zeros((len(env_ids), 3), device=self.device),
        ), dim=-1)
        obs2_new_state = torch.cat((
            obs2_new_pos,
            self.obs_std_orientation.repeat(len(env_ids), 1),
            torch.zeros((len(env_ids), 3), device=self.device),
            torch.zeros((len(env_ids), 3), device=self.device),
        ), dim=-1)
        obs3_new_state = torch.cat((
            obs3_new_pos,
            self.obs_std_orientation.repeat(len(env_ids), 1),
            torch.zeros((len(env_ids), 3), device=self.device),
            torch.zeros((len(env_ids), 3), device=self.device),
        ), dim=-1)
        self.obstacle1.write_root_state_to_sim(obs1_new_state, env_ids)
        self.obstacle2.write_root_state_to_sim(obs2_new_state, env_ids)
        self.obstacle3.write_root_state_to_sim(obs3_new_state, env_ids)


    # Ausiliary functions
    def compute_ee_pos(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        # hand world pose
        hand_pos_w = self.robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_quat_w = self.robot.data.body_quat_w[env_ids, self.hand_link_idx]
        hand_quat_w = hand_quat_w/(hand_quat_w.norm(dim=-1, keepdim=True) + 1e-8)
        # fingers pos
        lf_pos_w = self.robot.data.body_pos_w[env_ids, self.left_finger_link_idx]
        rf_pos_w = self.robot.data.body_pos_w[env_ids, self.right_finger_link_idx]
        midpoint_w = 0.5 * (lf_pos_w + rf_pos_w)
        #midpoint between fingers in local frame
        midpoint_l = self.rotate_vec_by_quat((midpoint_w - hand_pos_w), self.quat_conjugate(hand_quat_w))
        # adding the offset on the forward axis
        midpoint_l = midpoint_l + self.ee_offset * self.gripper_fwd_axis
        # back in world frame
        ee_offset_w = self.rotate_vec_by_quat(midpoint_l, hand_quat_w)
        self.robot_grasp_pos[env_ids] = ee_offset_w + hand_pos_w



    # def compute_probe_points_pos(self, joints_link_ids: torch.Tensor | None = None):
    #     if joints_link_ids is None:
    #         joints_link_ids = self.frame_link_ids

    #     p2 = self.robot.data.body_pos_w[:, self.frame_link_2_idx]
    #     p4 = self.robot.data.body_pos_w[:, self.frame_link_4_idx]
    #     p6 = self.robot.data.body_pos_w[:, self.frame_link_6_idx]
    #     p7 = self.robot.data.body_pos_w[:, self.hand_link_idx]
    #     pg = self.robot_grasp_pos

    #     def lerp_points(p1: torch.Tensor, p2: torch.Tensor, ts: list[float]) -> torch.Tensor:
    #         p1 = p1.unsqueeze(1)
    #         p2 = p2.unsqueeze(1)
    #         t  = torch.tensor(ts, device=self.device, dtype=p1.dtype).view(1, -1, 1)
    #         return (1.0 - t)*p1 + t*p2
        
    #     seg_24 = lerp_points(p2, p4, [0.25, 0.5, 0.75])
    #     seg_46 = lerp_points(p4, p6, [0.25, 0.5, 0.75])
    #     seg_67 = lerp_points(p6, p7, [0.33, 0.66])

    #     sx, sy, sz = 0.04, 0.04, 0.03
    #     tip = 0.22

    #     hand_offsets_local = torch.tensor(
    #         [
    #             [ +sx, 0.0, 0.0],
    #             [ -sx, 0.0, 0.0],
    #             [ 0.0, +sy, 0.0],
    #             [ 0.0, -sy, 0.0],
    #             [ 0.0, 0.0, +sz],
    #             [ 0.0, 0.0, -sz],
    #             [ 0.0, 0.0, +tip],  # extra forward tip
    #         ],
    #         device=self.device,
    #         dtype=p7.dtype,
    #     )  # (7,3)

    #     hand_quat = self.robot.data.body_quat_w[:, self.hand_link_idx]  # (N,4) [w,x,y,z] in IsaacLab
    #     hand_quat = hand_quat / (hand_quat.norm(dim=-1, keepdim=True) + 1e-8)

    #     # rotate offsets to world: (N,7,3)
    #     offs = hand_offsets_local.unsqueeze(0).expand(self.num_envs, -1, -1)
    #     offs_w = self.rotate_vec_by_quat(offs, hand_quat.unsqueeze(1).expand(-1, offs.shape[1], -1))
    #     hand_pts = p7.unsqueeze(1) + offs_w  # (N,7,3)

    #     # ---- stack everything into (N,20,3) ----

    #     anchors = torch.stack([p2, p4, p6, p7, pg], dim=1)

    #     probes_w = torch.cat(
    #         [
    #             anchors,        # (N,5,3)
    #             seg_24,         # (N,3,3)
    #             seg_46,         # (N,3,3)
    #             seg_67,         # (N,2,3)
    #             hand_pts,       # (N,7,3)
    #         ],
    #         dim=1,
    #     )
    #     return probes_w
    def compute_probe_points_pos(self, joints_link_ids: torch.Tensor | None = None):
        if joints_link_ids is None:
            joints_link_ids = self.frame_link_ids

        p2 = self.robot.data.body_pos_w[:, self.frame_link_2_idx]
        p4 = self.robot.data.body_pos_w[:, self.frame_link_4_idx]
        p6 = self.robot.data.body_pos_w[:, self.frame_link_6_idx]
        p7 = self.robot.data.body_pos_w[:, self.hand_link_idx]
        pg = self.robot_grasp_pos

        def lerp_points(p1: torch.Tensor, p2: torch.Tensor, ts: list[float]) -> torch.Tensor:
            p1 = p1.unsqueeze(1)
            p2 = p2.unsqueeze(1)
            t  = torch.tensor(ts, device=self.device, dtype=p1.dtype).view(1, -1, 1)
            return (1.0 - t)*p1 + t*p2
        
        seg_24 = lerp_points(p2, p4, [0.25, 0.75])
        seg_46 = lerp_points(p4, p6, [0.25, 0.75])
        seg_67 = lerp_points(p6, p7, [0.33, 0.66])

        sx, sy, sz = 0.04, 0.04, 0.03
        tip = 0.21

        hand_offsets_local = torch.tensor(
            [
                [ +sx, 0.0, 0.0],
                [ -sx, 0.0, 0.0],
                [ 0.0, +sy, 0.0],
                [ 0.0, -sy, 0.0],
                [ 0.0, 0.0, +sz],
                [ 0.0, 0.0, -sz],
                [ 0.0, 0.0, +tip],  # extra forward tip
            ],
            device=self.device,
            dtype=p7.dtype,
        )  # (7,3)

        hand_quat = self.robot.data.body_quat_w[:, self.hand_link_idx]  # (N,4) [w,x,y,z] in IsaacLab
        hand_quat = hand_quat / (hand_quat.norm(dim=-1, keepdim=True) + 1e-8)

        # rotate offsets to world: (N,7,3)
        offs = hand_offsets_local.unsqueeze(0).expand(self.num_envs, -1, -1)
        offs_w = self.rotate_vec_by_quat(offs, hand_quat.unsqueeze(1).expand(-1, offs.shape[1], -1))
        hand_pts = p7.unsqueeze(1) + offs_w  # (N,7,3)
        # Two fixed side probe points on the gripper body
        pg_body = self.robot.data.body_pos_w[:, self.gripper_body_idx]
        qg_body = self.robot.data.body_quat_w[:, self.gripper_body_idx]
        qg_body = qg_body / (qg_body.norm(dim=-1, keepdim=True) + 1e-8)
        side_y = 0.05      # left/right distance from gripper center
        side_x = 0.000      # vertical/local offset
        side_z = 0.085      # forward offset from panda_link7 toward the gripper body

        gripper_side_offsets_local = torch.tensor(
            [
                [ side_x, side_y, side_z],
                [ side_x, -side_y, side_z],
            ],
            device=self.device,
            dtype=p7.dtype,
        )  # (2, 3)

        side_offs = gripper_side_offsets_local.unsqueeze(0).expand(self.num_envs, -1, -1)

        side_offs_w = self.rotate_vec_by_quat(
            side_offs,
            qg_body.unsqueeze(1).expand(-1, side_offs.shape[1], -1)
        )

        gripper_side_pts = pg_body.unsqueeze(1) + side_offs_w  # (N,2,3)

        # ---- stack everything into (N,20,3) ----

        anchors = torch.stack([p2, p4, p6, p7, pg], dim=1)

        probes_w = torch.cat(
            [
                anchors,        # (N,5,3)
                seg_24,         # (N,3,3)
                seg_46,         # (N,3,3)
                seg_67,         # (N,2,3)
                hand_pts,       # (N,7,3)
                gripper_side_pts
            ],
            dim=1,
        )
        probe_points_flat = probes_w.reshape(-1, 3)
        # self.probe_marker.visualize(probe_points_flat.detach().cpu())
        #self.tip_marker.visualize(hand_pts[:, -1, :].detach().cpu())
        self.hand_tip_z = hand_pts[:, -1, -1]
        return probes_w
    
    
    def compute_min_obstacle_distance(self, robot_probes_vec: torch.Tensor):
        """
        Compute clearance between robot probe points and a finite vertical cylinder.
        Clearance > 0  : separated
        Clearance = 0  : touching (according to the probe/link-radius model)
        Clearance < 0  : penetration / unsafe overlap

        robot_probes_vec: (N, P, 3)
        returns:
            min_clearance: (N,)
            clearance:     (N, P)
        """
        obstacle1_pos_w = self.obstacle1.data.root_pos_w  # (N, 3)
        obstacle2_pos_w = self.obstacle2.data.root_pos_w
        obstacle3_pos_w = self.obstacle3.data.root_pos_w

        clearance1 = self.compute_dist_from_obstacle(robot_probes_vec, obstacle1_pos_w)
        clearance2 = self.compute_dist_from_obstacle(robot_probes_vec, obstacle2_pos_w)
        clearance3 = self.compute_dist_from_obstacle(robot_probes_vec, obstacle3_pos_w)

        clearance_all = torch.stack([clearance1, clearance2, clearance3], dim=-1)

        clearance = clearance_all.min(dim=-1).values
        min_clearance = clearance.min(dim=-1).values

        return min_clearance, clearance

        
    
    def compute_dist_from_obstacle(self, robot_probes_vec : torch.Tensor, obstacle_pos_w : torch.Tensor):

        # Cylinder geometry
        cyl_radius = self.cfg.obstacle_radius
        link_radius = self.cfg.link_radius
        cyl_half_h = 0.5 * self.cfg.obstacle_height

        cx = obstacle_pos_w[:, 0].unsqueeze(1)  # (N,1)
        cy = obstacle_pos_w[:, 1].unsqueeze(1)
        cz = obstacle_pos_w[:, 2].unsqueeze(1)

        px = robot_probes_vec[:, :, 0]
        py = robot_probes_vec[:, :, 1]
        pz = robot_probes_vec[:, :, 2]

        # Radial distance in XY from cylinder axis
        dx = px - cx
        dy = py - cy
        radial_dist = torch.sqrt(dx * dx + dy * dy + 1e-8) # (N,P)

        # Signed radial distance from infinite cylinder surface
        radial_out = radial_dist - cyl_radius  # >0 outside, <0 inside projected radius
        # Vertical distance from finite cylinder slab [-half_h, +half_h]
        dz = torch.abs(pz - cz) - cyl_half_h   # >0 above/below caps, <0 within height

        # Distance to finite cylinder surface (continuous)
        #
        # Standard SDF-style construction for a finite vertical cylinder:
        # outside both radially and vertically -> euclidean distance to edge/corner
        # outside radially only -> radial distance
        # outside vertically only -> vertical distance
        # inside both -> penetration = max(radial_out, dz) <= 0
        outside_radial = torch.clamp(radial_out, min=0.0)
        outside_vertical = torch.clamp(dz, min=0.0)

        outside_dist = torch.sqrt(outside_radial**2 + outside_vertical**2 + 1e-8)
        inside_dist = torch.clamp(torch.maximum(radial_out, dz), max=0.0)

        sdf_cylinder = outside_dist + inside_dist  # signed distance to cylinder surface

        # Clearance including link/probe thickness
        clearance = sdf_cylinder - link_radius
        return clearance

    def quat_conjugate(self, q: torch.Tensor):
        w, x, y, z = q.unbind(-1)
        q_conj = torch.stack((w, -x, -y, -z), dim=-1)
        return q_conj
    
    def rotate_vec_by_quat(self, v: torch.Tensor, q: torch.Tensor):  # v: (...,3), q: (...,4) [w,x,y,z]
        w, x, y, z = q.unbind(-1)
        qv = torch.stack((x, y, z), dim=-1)
        t  = 2.0 * torch.cross(qv, v, dim=-1)
        return v + w.unsqueeze(-1)*t + torch.cross(qv, t, dim=-1)
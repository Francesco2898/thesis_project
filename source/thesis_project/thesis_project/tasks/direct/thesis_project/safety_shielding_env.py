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
        self.hand_link_idx = self.robot.find_bodies("panda_link7")[0][0]
        self.frame_link_2_idx = self.robot.find_bodies("panda_link2")[0][0]
        self.frame_link_4_idx = self.robot.find_bodies("panda_link4")[0][0]
        self.frame_link_6_idx = self.robot.find_bodies("panda_link6")[0][0]
        self.frame_link_ids   = torch.tensor(
            [self.frame_link_2_idx, self.frame_link_4_idx, self.frame_link_6_idx, self.hand_link_idx],
            device=self.device, dtype=torch.long
        )
        
    
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)

        # Obstacle configuration
        obstacle_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Obstacle",
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
                pos= (0.4, 0.0, self.cfg.table_height + 0.5*self.cfg.obstacle_height),
                rot= (1.0, 0.0, 0.0, 0.0),
            ),
            debug_vis=True
        )
        self.obstacle = RigidObject(obstacle_cfg)

        # adding the components to the scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["obstacle"] = self.obstacle
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self.actions = self.actions[:, self.arm_ids]
        tau_scale = 0.2
        self.tau_cmd = self.actions * self.tau_max[0, self.arm_ids] * tau_scale
        self.tau_cmd = torch.clamp(self.tau_cmd, -self.tau_max[0, self.arm_ids], self.tau_max[0, self.arm_ids])

    
    def _apply_action(self) -> None:
        # Torque control
        self.robot.set_joint_effort_target(self.tau_cmd, joint_ids=self.arm_ids)
    
    def _get_observations(self):
        joint_pos = self.robot.data.joint_pos[:, self.arm_ids]
        joint_vel = self.robot.data.joint_vel[:, self.arm_ids]
        joint_pos_norm = 2.0 * (joint_pos - self.robot_dof_lower_limit[self.arm_ids])/(self.robot_dof_upper_limit[self.arm_ids] - self.robot_dof_lower_limit[self.arm_ids]) - 1.0
        joint_vel_norm = joint_vel * self.cfg.dof_velocity_scale
        q_err_norm = (self.q_goal - joint_pos)/self.joint_range[self.arm_ids]

        obs = torch.hstack((joint_pos_norm,
                            joint_vel_norm,
                            q_err_norm,
        ))
        obs = torch.nan_to_num(obs, nan=0, posinf=1e6, neginf=-1e6)
        observations = {"policy": torch.clamp(obs, -5.0, 5.0)}
        return observations
    
    def _get_rewards(self):
        joint_pos = self.robot.data.joint_pos[:, self.arm_ids]
        q_err_norm = (self.q_goal - joint_pos)/self.joint_range[self.arm_ids]
        # q_dist = torch.norm(q_err_norm, dim=-1)
        q_err = torch.norm(q_err_norm, dim=-1)
        prev_q_err = torch.norm(self.prev_q_err, dim=-1)
        reward_progress = (prev_q_err - q_err) * 10.0

        q_abs = torch.abs(q_err_norm)
        q_worst = torch.max(q_abs, dim=-1)
        mean_sq = torch.mean(q_abs**2, dim=-1)
        reward_joint = torch.exp(-10.0 * mean_sq)
        # reward_for_joint = 1.0/(1.0 + (q_abs)**2)
        # reward_for_joint = torch.where(q_dist < 0.2, 2.0 * reward_for_joint, reward_for_joint)
        # reward_joint = torch.sum(reward_for_joint, dim=-1)*0.1

        # actions penalty
        penalty_action = torch.sum(self.actions**2, dim=-1)
        penalty_action = penalty_action * self.cfg.action_penalty_scale
        # joint_vel = self.robot.data.joint_vel
        # penalty_action = torch.norm(joint_vel, dim=-1) * self.cfg.action_penalty_scale

        reward = (reward_joint
                  - penalty_action)
        if self.global_step % 100 == 0:
            self.writer.add_scalar("geom/q_dist", q_err.mean().item(), self.global_step)
            for i in range(len(self.arm_ids)):
                self.writer.add_scalar(f"geom/q_err_norm_joint_{i}", q_err_norm[:, i].abs().mean().item(), self.global_step)
            self.writer.add_scalar("reward/reward_joint", reward_joint.mean().item(), self.global_step)
            self.writer.add_scalar("reward/reward_progress", reward_progress.mean().item(), self.global_step)
            self.writer.add_scalar("reward/total", reward.mean().item(), self.global_step)
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

        joint_pos = self.robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self.robot.num_joints),
            device= self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limit, self.robot_dof_upper_limit)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.tau_cmd[env_ids] = 0.0
        self.q_goal[env_ids] = sample_uniform(
            self.robot_dof_lower_limit[self.arm_ids],
            self.robot_dof_upper_limit[self.arm_ids],
            (len(env_ids), len(self.arm_ids)),
            device=self.device
        )
        self.prev_q_err[env_ids] = (self.q_goal[env_ids] - joint_pos[:, self.arm_ids])/self.joint_range[self.arm_ids]


    # Ausiliary functions
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
    #     self.tip_marker.visualize(hand_pts[:, -1, :].detach().cpu())
    #     return probes_w
    
    # def compute_min_obstacle_distance(self, robot_probes_vec: torch.Tensor ):
    #     obstacle_pos_w = self.obstacle.data.root_pos_w

    #     dxy = robot_probes_vec[:, :, :2] - obstacle_pos_w[:, None, :2]
    #     z_probe  = robot_probes_vec[:, :, 2]
    #     z_th = (obstacle_pos_w[:, 2] + 0.5*self.cfg.obstacle_height + self.cfg.safe_margin).unsqueeze(1)
    #     dist = torch.sqrt((dxy**2).sum(dim=-1) + 1e-8)

    #     clearance = dist - (self.cfg.obstacle_radius + self.cfg.link_radius)
    #     clearance = torch.where(z_probe <= z_th, clearance, torch.full_like(clearance, self.cfg.safe_margin))
    #     min_clearance = torch.min(clearance, dim=-1).values
    #     return min_clearance, clearance
    
    # def compute_min_obstacle_distance(self, robot_probes_vec: torch.Tensor):
    #     """
    #     Compute clearance between robot probe points and a finite vertical cylinder.
    #     Clearance > 0  : separated
    #     Clearance = 0  : touching (according to the probe/link-radius model)
    #     Clearance < 0  : penetration / unsafe overlap

    #     robot_probes_vec: (N, P, 3)
    #     returns:
    #         min_clearance: (N,)
    #         clearance:     (N, P)
    #     """
    #     obstacle_pos_w = self.obstacle.data.root_pos_w  # (N, 3)

    #     # Cylinder geometry
    #     cyl_radius = self.cfg.obstacle_radius
    #     link_radius = self.cfg.link_radius
    #     cyl_half_h = 0.5 * self.cfg.obstacle_height

    #     cx = obstacle_pos_w[:, 0].unsqueeze(1)  # (N,1)
    #     cy = obstacle_pos_w[:, 1].unsqueeze(1)
    #     cz = obstacle_pos_w[:, 2].unsqueeze(1)

    #     px = robot_probes_vec[:, :, 0]
    #     py = robot_probes_vec[:, :, 1]
    #     pz = robot_probes_vec[:, :, 2]

    #     # Radial distance in XY from cylinder axis
    #     dx = px - cx
    #     dy = py - cy
    #     radial_dist = torch.sqrt(dx * dx + dy * dy + 1e-8)  # (N,P)

    #     # Signed radial distance from infinite cylinder surface
    #     radial_out = radial_dist - cyl_radius  # >0 outside, <0 inside projected radius

    #     # Vertical distance from finite cylinder slab [-half_h, +half_h]
    #     dz = torch.abs(pz - cz) - cyl_half_h   # >0 above/below caps, <0 within height

    #     # Distance to finite cylinder surface (continuous)
    #     #
    #     # Standard SDF-style construction for a finite vertical cylinder:
    #     # outside both radially and vertically -> euclidean distance to edge/corner
    #     # outside radially only -> radial distance
    #     # outside vertically only -> vertical distance
    #     # inside both -> penetration = max(radial_out, dz) <= 0
    #     outside_radial = torch.clamp(radial_out, min=0.0)
    #     outside_vertical = torch.clamp(dz, min=0.0)

    #     outside_dist = torch.sqrt(outside_radial**2 + outside_vertical**2 + 1e-8)
    #     inside_dist = torch.clamp(torch.maximum(radial_out, dz), max=0.0)

    #     sdf_cylinder = outside_dist + inside_dist  # signed distance to cylinder surface

    #     # Clearance including link/probe thickness
    #     clearance = sdf_cylinder - link_radius

    #     min_clearance = clearance.min(dim=-1).values
    #     return min_clearance, clearance
    
    # def rotate_vec_by_quat(self, v: torch.Tensor, q: torch.Tensor):  # v: (...,3), q: (...,4) [w,x,y,z]
    #     w, x, y, z = q.unbind(-1)
    #     qv = torch.stack((x, y, z), dim=-1)
    #     t  = 2.0 * torch.cross(qv, v, dim=-1)
    #     return v + w.unsqueeze(-1)*t + torch.cross(qv, t, dim=-1)
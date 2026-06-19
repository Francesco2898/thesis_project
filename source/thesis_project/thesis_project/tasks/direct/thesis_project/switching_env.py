from __future__ import annotations
import torch

from .pick_place_env import ThesisProjectEnv
from .switching_env_cfg import SwitchingEnvCfg

class SwitchingEnv(ThesisProjectEnv):
    cfg: SwitchingEnvCfg

    def __init__(self, cfg: SwitchingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.arm_ids = torch.arange(self.robot.num_joints-2, device=self.device, dtype=torch.long)
        self.joint_range = torch.clamp(self.robot_dof_upper_limit - self.robot_dof_lower_limit, min=1e-6)
        self.q_goal = torch.tensor(self.cfg.q_goal_safe_cfg,
                                   device=self.device,
                                   dtype=torch.float32).view(1, -1).repeat(self.num_envs, 1)
        self.min_dist_buf = torch.full(
            (self.num_envs,),
            self.cfg.safe_margin,
            device=self.device,
        )

    def get_task_obs_for_inference(self) -> torch.Tensor:
        return self._get_observations()["policy"]
    
    def get_safety_obs_for_inference(self) -> torch.Tensor:
         joint_pos = self.robot.data.joint_pos[:, self.arm_ids]
         joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limit[self.arm_ids], self.robot_dof_upper_limit[self.arm_ids])
         joint_vel = self.robot.data.joint_vel[:, self.arm_ids]
         joint_pos_norm = 2.0 * (joint_pos - self.robot_dof_lower_limit[self.arm_ids])/(self.robot_dof_upper_limit[self.arm_ids] - self.robot_dof_lower_limit[self.arm_ids]) - 1.0
         joint_vel_norm = joint_vel * self.cfg.safety_dof_velocity_scale
         q_err_norm = (self.q_goal - joint_pos)/self.joint_range[self.arm_ids]
         # Recompute end-effector/probes/clearance
         self.compute_ee_pos()
         self.probes_w_buf = self.compute_probe_points_pos()
         _, self.clear_vec_buf = self.compute_min_obstacle_distance(self.probes_w_buf)
         scale = 0.2
         clear_obs = torch.tanh(self.clear_vec_buf/scale)

         obs = torch.hstack((joint_pos_norm,
                             joint_vel_norm,
                             q_err_norm,
                             clear_obs))
         obs = torch.nan_to_num(obs, nan=0, posinf=1e6, neginf=-1e6)
         return torch.clamp(obs, -5.0, 5.0)
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.compute_intermediate_values()
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self.probes_w_buf = self.compute_probe_points_pos()
        self.min_dist_buf, self.clear_vec_buf = self.compute_min_obstacle_distance(self.probes_w_buf)
        # self.saving_q_safe(self.min_dist_buf)

        # Arm actions
        arm_actions = self.actions
        targets_arm = (self.robot_dof_targets[:, self.arm_ids]
                       + self.robot_dof_speed_scales[self.arm_ids]
                       * self.dt
                       * arm_actions
                       * self.cfg.action_scale_arm)
        
        self.robot_dof_targets[:, self.arm_ids] = torch.clamp(targets_arm,
                                                              self.robot_dof_lower_limit[self.arm_ids],
                                                              self.robot_dof_upper_limit[self.arm_ids])

        # Fingers actions
        # gripper_cmd = self.actions[:, -2:-1].clamp(-1.0, 1.0)
        # gripper_cmd = gripper_cmd.expand(-1, 2)
        OPEN = 1.0
        CLOSE = -1.0

        dist_pick = torch.norm(self.object_grasp_pos_w - self.robot_grasp_pos, dim=-1)
        dist_place = torch.norm(self.place_pos_w - self.target_obj.data.root_pos_w, dim=-1)

        pick_cnd = dist_pick < self.r_in
        place_cnd = dist_place < self.r_in
        pick_task = ~self.has_object
        place_task = self.has_object

        close_gripper = pick_task & pick_cnd
        open_gripper = (place_task & place_cnd) | (pick_task & (dist_pick > self.r_in))

        gs = self.gripper_state
        gs = torch.where(close_gripper, torch.zeros_like(gs), gs)
        gs = torch.where(open_gripper, torch.ones_like(gs), gs)
        self.gripper_state = gs

        cmd = torch.where(self.gripper_state > 0.5,
                          torch.full_like(self.gripper_state, OPEN),
                          torch.full_like(self.gripper_state, CLOSE))
        cmd = cmd.unsqueeze(-1).expand(-1, 2)
        
        targets_gripper = (self.robot_dof_targets[:, self.finger_joints_ids]
                           + self.robot_dof_speed_scales[self.finger_joints_ids]
                           * self.dt
                           * cmd
                           * 0.8)
        
        self.robot_dof_targets[:, self.finger_joints_ids] = torch.clamp(targets_gripper,
                                                                     self.robot_dof_lower_limit[self.finger_joints_ids],
                                                                     self.robot_dof_upper_limit[self.finger_joints_ids])
    
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)

        # self.q_goal[env_ids] = self.q_safe_buff[env_ids].clone()
        self.min_dist_buf[env_ids] = self.cfg.safe_margin

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
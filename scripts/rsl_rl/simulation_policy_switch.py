import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Simulation with switching policies")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task")

parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# exported policies
parser.add_argument("--task_policy_path", type=str, required=True, help="Path to exported main task policy.pt.")
parser.add_argument("--safety_policy_path", type=str, required=True, help="Path to exported safety policy.pt.")
# switching thresholds
parser.add_argument("--d_enter", type=float, default=0.06, help="Clearance threshold to enter safety mode.")
parser.add_argument("--d_exit", type=float, default=0.12, help="Clearance threshold to exit safety mode.")
parser.add_argument("--q_threshold", type=float, default=0.15, help="Safe-goal error threshold to exit safety mode.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable camera to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import thesis_project.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
):
    """Play with switching between task and safety policies."""

    # Override configurations with CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Create Isaac Lab environment
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # Convert multi-agent env to single-agent env if needed
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Optional video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": "videos/play_with_switch",
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap environment for RSL-RL-style stepping/action clipping
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    base_env = env.unwrapped
    device = base_env.device
    dt = base_env.step_dt

    # Load exported TorchScript policies
    print(f"[INFO] Loading task policy from: {args_cli.task_policy_path}")
    task_policy = torch.jit.load(args_cli.task_policy_path, map_location=device).to(device)
    task_policy.eval()

    print(f"[INFO] Loading safety policy from: {args_cli.safety_policy_path}")
    safety_policy = torch.jit.load(args_cli.safety_policy_path, map_location=device).to(device)
    safety_policy.eval()

    # Initialize mode: 0 = task policy, 1 = safety policy
    mode = torch.zeros(base_env.num_envs, dtype=torch.long, device=device)

    # Reset/get initial observations
    obs, _ = env.get_observations()
    timestep = 0

    # Main simulation loop
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # Build observations
            obs_task = base_env.get_task_obs_for_inference()

            # Safety switch condition
            base_env.compute_intermediate_values()
            base_env.probes_w_buf = base_env.compute_probe_points_pos()
            base_env.min_dist_buf, base_env.clear_vec_buf = base_env.compute_min_obstacle_distance(base_env.probes_w_buf)
            min_clearance = base_env.min_dist_buf

            q = base_env.robot.data.joint_pos[:, base_env.arm_ids]
            dq = base_env.robot.data.joint_vel[:, base_env.arm_ids]
            dq_norm = torch.norm(dq, dim=-1)
            joint_range = base_env.joint_range[base_env.arm_ids]

            enter_safe = (mode == 0) & (min_clearance < args_cli.d_enter)
            if enter_safe.any():
                # base_env.q_goal[enter_safe] = base_env.q_safe_buff[enter_safe].clone()
                # base_env.robot_dof_targets[enter_safe, :-2] = base_env.robot.data.joint_pos[enter_safe, :-2]
                mode[enter_safe] = 1

            obs_safe = base_env.get_safety_obs_for_inference()

            # Compute both candidate actions
            action_task = task_policy(obs_task)
            action_safe_arm = safety_policy(obs_safe)

            action_safe = torch.zeros_like(action_task)
            action_safe[:, base_env.arm_ids] = action_safe_arm
            action_safe[:, -2:] = action_task[:, -2:]

            q_goal = base_env.q_goal

            q_err_safe = torch.norm((q_goal - q) / joint_range, dim=-1)

            # exit_safe = (
            #     (mode == 1)
            #     & (min_clearance > args_cli.d_exit)
            #     & (q_err_safe < args_cli.q_threshold)
            #     & (dq_norm < 0.15)
            # )
            # if exit_safe.any():
            #     base_env.robot_dof_targets[exit_safe, :-2] = base_env.robot.data.joint_pos[exit_safe, :-2]
            #     mode[exit_safe] = 0

            use_safe = mode == 1
            # Select final action
            actions = torch.where(
                use_safe.unsqueeze(-1),
                action_safe,
                action_task,
            )

            if timestep % 20 == 0:
                print(
                    "[DEBUG]",
                    "mode:", mode[0].item(),
                    "min_clearance:", min_clearance[0].item(),
                    "d_enter:", args_cli.d_enter,
                    "q_err_safe:", q_err_safe[0].item(),
                    "dq_norm:", dq_norm[0].item(),
                )
                # print("q_safe_buff:", base_env.q_safe_buff[0].detach().cpu().numpy())
                # print("q_goal:", base_env.q_goal[0].detach().cpu().numpy())

            # Step environment
            obs, _, dones, _ = env.step(actions)
            if dones.any():
                reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)

                mode[reset_ids] = 0
                # base_env.q_goal[reset_ids] = base_env.q_safe_buff[reset_ids].clone()
            timestep += 1

        if args_cli.video:
            if timestep == args_cli.video_length:
                break

        # Optional real-time delay
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

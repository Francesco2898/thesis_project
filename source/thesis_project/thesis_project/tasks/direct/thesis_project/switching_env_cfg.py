from isaaclab.utils import configclass

from .thesis_project_env_cfg import ThesisProjectEnvCfg


@configclass
class SwitchingEnvCfg(ThesisProjectEnvCfg):
    # - space definition
    action_space = 9
    observation_space = 46
    state_space = 0
    
    # definition of the safe condition for shielding
    q_goal_safe_cfg = [
        0.2563881650718601,
        0.4500463124520452,
        0.3697866797164865,
        -1.9843721466291926,
        -0.23336613369861603,
        2.393948550913406,
        1.5501551389393688,
    ]

    safety_dof_velocity_scale = 0.3
    safety_clearance_scale = 0.2
    action_scale = 0.35
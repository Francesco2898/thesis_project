import numpy as np

path = "policy_debug_samples.npz"
data = np.load(path)

obs = data["obs"]
actions = data["actions"]

N = min(5, obs.shape[0])
OBS_DIM = obs.shape[1]
ACT_DIM = actions.shape[1]

print(f"// Loaded from {path}")
print(f"// obs shape = {obs.shape}")
print(f"// actions shape = {actions.shape}")
print()

print(f"const std::array<std::array<float, {OBS_DIM}>, {N}> obs_debug_seq = {{ {{")
for i in range(N):
    print("  {{")
    print("    " + ", ".join(f"{v:.8g}f" for v in obs[i]))
    print("  }},")
print("} };")

print()

print(f"const std::array<std::array<float, {ACT_DIM}>, {N}> py_action_ref = {{ {{")
for i in range(N):
    print("  {{")
    print("    " + ", ".join(f"{v:.8g}f" for v in actions[i]))
    print("  }},")
print("} };")
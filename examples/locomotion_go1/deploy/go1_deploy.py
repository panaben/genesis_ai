#!/usr/bin/env python3
"""
Go1 real-robot deployment script.

Loads a trained policy (model_N.pt + cfgs.pkl) and runs it on a physical
Unitree Go1 via unitree_legged_sdk over UDP at 50 Hz.

Prerequisites
-------------
1. Build the SDK Python wrapper (do this once on the deploy PC):

       cd <repo_root>/thirdparty/unitree_legged_sdk
       mkdir -p build && cd build
       cmake -DPYTHON_BUILD=TRUE ..
       make

2. Connect the deploy PC to Go1 (Ethernet, 192.168.123.x subnet).

3. Set Go1 to DAMPING mode from the remote before running this script.

Usage
-----
    sudo python go1_deploy.py --model_dir logs/go1-walking --ckpt 100 \\
        --vx 0.3 --vy 0.0 --wz 0.0 --log

Keyboard control during walk:
    Ctrl+C  →  stop and enter damping mode

Motor index mapping (matches go1_train.py joint_names order):
    0:FR_hip  1:FR_thigh  2:FR_calf
    3:FL_hip  4:FL_thigh  5:FL_calf
    6:RR_hip  7:RR_thigh  8:RR_calf
    9:RL_hip 10:RL_thigh 11:RL_calf
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import signal
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

# ---------------------------------------------------------------------------
# unitree_legged_sdk Python wrapper
# Resolved relative to this script: <repo>/thirdparty/unitree_legged_sdk/
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SDK_ROOT = _REPO_ROOT / "thirdparty" / "unitree_legged_sdk"

def _find_sdk_lib() -> Path:
    for arch in ("amd64", "arm64"):
        candidate = _SDK_ROOT / "lib" / "python" / arch
        if (candidate / "robot_interface.so").exists():
            return candidate
    raise FileNotFoundError(
        f"robot_interface.so not found under {_SDK_ROOT}/lib/python/.\n"
        "Build the SDK: cd thirdparty/unitree_legged_sdk/build && "
        "cmake -DPYTHON_BUILD=TRUE .. && make"
    )

sys.path.insert(0, str(_find_sdk_lib()))
import robot_interface as sdk  # noqa: E402  (must come after sys.path update)

# ---------------------------------------------------------------------------
# rsl_rl version check (matches go1_eval.py)
# ---------------------------------------------------------------------------
try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as _e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from _e

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

# ---------------------------------------------------------------------------
# Robot-specific constants  (must match go1_train.py get_cfgs())
# ---------------------------------------------------------------------------

ROBOT_IP   = "192.168.123.10"
ROBOT_PORT = 8007
LOCAL_PORT = 8090

NUM_MOTORS     = 12
CONTROL_DT     = 0.02   # 50 Hz — must match env_cfg["dt"]

# Standup interpolation phase
STANDUP_DURATION_S = 2.0
STANDUP_KP = 5.0
STANDUP_KD = 1.0

# Walking phase PD gains (must match env_cfg["kp"] / ["kd"])
DEPLOY_KP = 20.0
DEPLOY_KD = 0.5

# Safe shutdown mode
DAMPING_KP = 0.0
DAMPING_KD = 2.0

# Emergency stop thresholds
EMERGENCY_ROLL_DEG  = 30.0
EMERGENCY_PITCH_DEG = 30.0

# Observation scaling (must match obs_cfg["obs_scales"])
_OBS_SCALE_ANG_VEL = 0.25
_OBS_SCALE_DOF_POS = 1.0
_OBS_SCALE_DOF_VEL = 0.05
# commands_scale = [lin_vel_scale, lin_vel_scale, ang_vel_scale]
_COMMANDS_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)

# Action mapping (must match env_cfg)
ACTION_SCALE = 0.25
CLIP_ACTIONS = 100.0

OBS_DIM = 45  # 3+3+3+12+12+12  — must match go1_env._update_observation()

# Default joint positions in joint_names order [FR, FL, RR, RL] × [hip, thigh, calf]
# Values from go1_train.py get_cfgs() "default_joint_angles"
DEFAULT_DOF_POS = np.array(
    [
        0.0,  0.8, -1.5,  # FR_hip, FR_thigh, FR_calf
        0.0,  0.8, -1.5,  # FL_hip, FL_thigh, FL_calf
        0.0,  1.0, -1.5,  # RR_hip, RR_thigh, RR_calf
        0.0,  1.0, -1.5,  # RL_hip, RL_thigh, RL_calf
    ],
    dtype=np.float32,
)

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _quat_rotate(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate vector v (3,) by unit quaternion q = [w, x, y, z].

    Uses Rodrigues' formula:
        v' = v + 2w (q_v × v) + 2 (q_v × (q_v × v))
    """
    w, x, y, z = q.astype(np.float64)
    q_v = np.array([x, y, z])
    v64 = v.astype(np.float64)
    return (v64 + 2.0 * w * np.cross(q_v, v64) + 2.0 * np.cross(q_v, np.cross(q_v, v64))).astype(np.float32)


def _projected_gravity(q_wxyz: np.ndarray) -> np.ndarray:
    """Return world gravity [0,0,-1] expressed in robot body frame.

    Equivalent to go1_env:
        inv_base_quat = inv_quat(base_quat)
        projected_gravity = transform_by_quat([0,0,-1], inv_base_quat)

    SDK quaternion convention: [w, x, y, z]  (same as Genesis).
    """
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    w, x, y, z = q_wxyz
    q_inv = np.array([w, -x, -y, -z], dtype=np.float32)  # conjugate = inverse for unit quat
    return _quat_rotate(gravity_world, q_inv)


def _euler_from_quat_deg(q_wxyz: np.ndarray) -> tuple[float, float, float]:
    """Return (roll, pitch, yaw) in degrees from quaternion [w, x, y, z]."""
    w, x, y, z = q_wxyz.astype(np.float64)
    roll  = math.degrees(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))
    yaw   = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return roll, pitch, yaw

# ---------------------------------------------------------------------------
# Minimal env stub for OnPolicyRunner initialisation (no Genesis required)
# ---------------------------------------------------------------------------

class _DeployEnv:
    """Satisfies the interface expected by rsl_rl >= 5.0 OnPolicyRunner.

    Only used to initialise the runner's networks; never stepped during deploy.
    """

    num_envs:   int = 1
    num_actions: int = NUM_MOTORS
    device: str = "cpu"

    def get_observations(self) -> TensorDict:
        return TensorDict({"policy": torch.zeros(1, OBS_DIM)}, batch_size=[1])

    def reset(self) -> TensorDict:
        return self.get_observations()

    def step(self, actions: torch.Tensor):
        obs = self.get_observations()
        return obs, torch.zeros(1), torch.zeros(1, dtype=torch.bool), {}


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def load_policy(model_dir: str, ckpt: int, train_cfg: dict) -> callable:
    """Load trained actor from checkpoint, return inference callable."""
    env = _DeployEnv()
    runner = OnPolicyRunner(env, train_cfg, model_dir, device="cpu")
    ckpt_path = os.path.join(model_dir, f"model_{ckpt}.pt")
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device="cpu")
    print(f"[deploy] Loaded checkpoint: {ckpt_path}")
    return policy


# ---------------------------------------------------------------------------
# SDK command helpers
# ---------------------------------------------------------------------------

def _set_motor_cmd(
    cmd: sdk.LowCmd,
    target_q: np.ndarray,
    kp: float,
    kd: float,
) -> None:
    """Fill LowCmd with position targets and PD gains for all motors."""
    for i in range(NUM_MOTORS):
        cmd.motorCmd[i].mode = 0x0A  # FOC / servo mode
        cmd.motorCmd[i].q    = float(target_q[i])
        cmd.motorCmd[i].dq   = 0.0
        cmd.motorCmd[i].tau  = 0.0
        cmd.motorCmd[i].Kp   = kp
        cmd.motorCmd[i].Kd   = kd


def _set_damping(cmd: sdk.LowCmd) -> None:
    """Set all motors to passive damping (safe shutdown posture)."""
    for i in range(NUM_MOTORS):
        cmd.motorCmd[i].mode = 0x0A
        cmd.motorCmd[i].q    = 0.0
        cmd.motorCmd[i].dq   = 0.0
        cmd.motorCmd[i].tau  = 0.0
        cmd.motorCmd[i].Kp   = DAMPING_KP
        cmd.motorCmd[i].Kd   = DAMPING_KD


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy Go1 locomotion policy on real robot.")
    parser.add_argument("--model_dir", type=str, default="logs/go1-walking",
                        help="Directory containing cfgs.pkl and model_N.pt")
    parser.add_argument("--ckpt", type=int, default=100,
                        help="Checkpoint index (e.g. 100 → model_100.pt)")
    parser.add_argument("--vx",  type=float, default=0.3,  help="Forward velocity command (m/s)")
    parser.add_argument("--vy",  type=float, default=0.0,  help="Lateral velocity command (m/s)")
    parser.add_argument("--wz",  type=float, default=0.0,  help="Yaw rate command (rad/s)")
    parser.add_argument("--log", action="store_true",       help="Enable data logging to CSV")
    parser.add_argument("--log_dir", type=str, default="deploy_logs",
                        help="Directory for CSV log output")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load configs and policy
    # ------------------------------------------------------------------
    cfgs_path = os.path.join(args.model_dir, "cfgs.pkl")
    with open(cfgs_path, "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    print(f"[deploy] Loading policy from {args.model_dir}/model_{args.ckpt}.pt ...")
    policy = load_policy(args.model_dir, args.ckpt, train_cfg)

    commands = np.array([args.vx, args.vy, args.wz], dtype=np.float32)
    print(f"[deploy] Commands: vx={args.vx} m/s  vy={args.vy} m/s  wz={args.wz} rad/s")

    # ------------------------------------------------------------------
    # Optional logger
    # ------------------------------------------------------------------
    logger = None
    if args.log:
        from go1_logger import DataLogger
        os.makedirs(args.log_dir, exist_ok=True)
        log_path = os.path.join(args.log_dir, f"run_{int(time.time())}.csv")
        logger = DataLogger(log_path)
        print(f"[deploy] Logging enabled → {log_path}")

    # ------------------------------------------------------------------
    # SDK initialisation
    # ------------------------------------------------------------------
    safe = sdk.Safety(sdk.LeggedType.Go1)
    udp  = sdk.UDP(sdk.LOWLEVEL, LOCAL_PORT, ROBOT_IP, ROBOT_PORT)
    cmd   = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)

    # ------------------------------------------------------------------
    # Signal handler for graceful Ctrl+C shutdown
    # ------------------------------------------------------------------
    running = True

    def _on_sigint(sig, frame):
        nonlocal running
        print("\n[deploy] Ctrl+C received — stopping after current step.")
        running = False

    signal.signal(signal.SIGINT, _on_sigint)

    # ------------------------------------------------------------------
    # Read initial joint positions (warm up UDP connection)
    # ------------------------------------------------------------------
    print("[deploy] Reading initial joint positions ...")
    for _ in range(10):
        udp.Recv()
        udp.GetRecv(state)
        time.sleep(0.002)

    q_init = np.array([state.motorState[i].q for i in range(NUM_MOTORS)], dtype=np.float32)
    print(f"[deploy] Initial q : {np.round(q_init, 3)}")
    print(f"[deploy] Default q : {np.round(DEFAULT_DOF_POS, 3)}")

    # ------------------------------------------------------------------
    # Standup phase: interpolate from q_init to DEFAULT_DOF_POS over 2 s
    # Uses low Kp/Kd for safe, slow movement.
    # ------------------------------------------------------------------
    input(
        "\n[deploy] *** Verify the robot is suspended or on flat ground. ***\n"
        "         Press Enter to begin STANDUP sequence ..."
    )

    standup_steps = int(STANDUP_DURATION_S / CONTROL_DT)
    print(f"[deploy] Standing up ({STANDUP_DURATION_S:.1f} s, {standup_steps} steps) ...")

    for step_i in range(standup_steps):
        t0 = time.monotonic()
        rate = (step_i + 1) / standup_steps
        target_q = q_init * (1.0 - rate) + DEFAULT_DOF_POS * rate

        _set_motor_cmd(cmd, target_q, STANDUP_KP, STANDUP_KD)
        safe.PowerProtect(cmd, state, 6)
        udp.SetSend(cmd)
        udp.Send()

        udp.Recv()
        udp.GetRecv(state)

        dt_used = time.monotonic() - t0
        remaining = CONTROL_DT - dt_used
        if remaining > 0:
            time.sleep(remaining)

    print("[deploy] Standup complete.")
    input("[deploy] >>> Press Enter to start WALKING policy ...")

    # ------------------------------------------------------------------
    # Main control loop  (50 Hz)
    #
    # Replicates go1_env.step() + _update_observation() exactly:
    #   - obs uses `actions`  (current step's computed action, not yet sent)
    #   - motor receives `last_actions`  (simulate_action_latency=True)
    # ------------------------------------------------------------------
    actions      = np.zeros(NUM_MOTORS, dtype=np.float32)  # current computed action (in obs)
    last_actions = np.zeros(NUM_MOTORS, dtype=np.float32)  # previous step's action (sent to motor)
    step_count   = 0
    overrun_count = 0

    print("[deploy] Walking policy active. Press Ctrl+C to stop.")

    while running:
        t0 = time.monotonic()

        # 1. Receive robot state
        udp.Recv()
        udp.GetRecv(state)

        # 2. Safety check — stop if robot is falling
        quat = np.array(state.imu.quaternion, dtype=np.float32)  # [w, x, y, z]
        roll_deg, pitch_deg, _ = _euler_from_quat_deg(quat)
        if abs(roll_deg) > EMERGENCY_ROLL_DEG or abs(pitch_deg) > EMERGENCY_PITCH_DEG:
            print(
                f"[deploy] EMERGENCY STOP: roll={roll_deg:.1f}° pitch={pitch_deg:.1f}°"
                f" exceeds limit ({EMERGENCY_ROLL_DEG}°/{EMERGENCY_PITCH_DEG}°)"
            )
            running = False
            break

        # 3. Build observation vector — must exactly match go1_env._update_observation()
        #
        #   obs = [base_ang_vel*0.25,   # 3  — IMU gyroscope (body frame)
        #          projected_gravity,    # 3  — gravity in body frame
        #          commands * scale,     # 3  — [vx*2, vy*2, wz*0.25]
        #          (dof_pos-default)*1,  # 12 — joint position offset
        #          dof_vel * 0.05,       # 12 — joint velocity
        #          actions]              # 12 — current step's computed action
        gyro    = np.array(state.imu.gyroscope, dtype=np.float32)            # [rad/s], body frame
        pg      = _projected_gravity(quat)                                    # body frame gravity
        dof_pos = np.array([state.motorState[i].q  for i in range(NUM_MOTORS)], dtype=np.float32)
        dof_vel = np.array([state.motorState[i].dq for i in range(NUM_MOTORS)], dtype=np.float32)

        obs_np = np.concatenate([
            gyro    * _OBS_SCALE_ANG_VEL,               # 3
            pg,                                          # 3
            commands * _COMMANDS_SCALE,                  # 3
            (dof_pos - DEFAULT_DOF_POS) * _OBS_SCALE_DOF_POS,  # 12
            dof_vel * _OBS_SCALE_DOF_VEL,               # 12
            actions,                                     # 12  ← current step's action (not yet sent)
        ])  # total: 45

        # 4. Policy inference
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0)          # [1, 45]
        obs_dict   = TensorDict({"policy": obs_tensor}, batch_size=[1])

        with torch.no_grad():
            new_actions_tensor = policy(obs_dict)

        new_actions = new_actions_tensor.cpu().numpy().squeeze(0).astype(np.float32)
        new_actions = np.clip(new_actions, -CLIP_ACTIONS, CLIP_ACTIONS)

        # 5. Compute motor target using LAST step's action
        #    (replicates simulate_action_latency=True from training)
        exec_actions  = last_actions
        target_dof_pos = exec_actions * ACTION_SCALE + DEFAULT_DOF_POS

        # 6. Send command to robot
        _set_motor_cmd(cmd, target_dof_pos, DEPLOY_KP, DEPLOY_KD)
        safe.PowerProtect(cmd, state, 6)
        udp.SetSend(cmd)
        udp.Send()

        # 7. Log state for System ID
        if logger is not None:
            tau_est = np.array([state.motorState[i].tauEst for i in range(NUM_MOTORS)], dtype=np.float32)
            logger.log(
                timestamp=time.monotonic(),
                dof_pos=dof_pos,
                dof_vel=dof_vel,
                tau_est=tau_est,
                imu_gyro=gyro,
                imu_quat=quat,
                commands=commands,
                actions=new_actions,
            )

        # 8. Advance buffers
        last_actions = actions.copy()
        actions      = new_actions
        step_count  += 1

        # 9. Timing — busy-wait not used; sleep for remaining dt
        dt_used   = time.monotonic() - t0
        remaining = CONTROL_DT - dt_used
        if remaining > 0:
            time.sleep(remaining)
        else:
            overrun_count += 1
            if overrun_count % 50 == 1:
                print(f"[deploy] Loop overrun: {-remaining * 1000:.1f} ms late  (step {step_count})")

    # ------------------------------------------------------------------
    # Shutdown — switch to passive damping
    # ------------------------------------------------------------------
    print(f"[deploy] Stopping after {step_count} steps — entering damping mode ...")
    _set_damping(cmd)
    udp.SetSend(cmd)
    udp.Send()
    time.sleep(0.5)

    if logger is not None:
        logger.close()

    print("[deploy] Done.")


if __name__ == "__main__":
    main()

"""
# Example usage (run as root for memory locking):
sudo python examples/locomotion_go1/deploy/go1_deploy.py \\
    --model_dir logs/go1-walking --ckpt 100 \\
    --vx 0.3 --log
"""

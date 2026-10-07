#!/usr/bin/env python3
"""
Go1 real-robot deployment script.

Loads a trained policy (model_N.pt + cfgs.pkl) and runs it on a physical
Unitree Go1 via unitree_legged_sdk over UDP at 50 Hz.

Prerequisites
-------------
1. Build the SDK Python wrapper (do this once on the deploy PC):

       cd <repo_root>/third_party/unitree_legged_sdk
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
# Resolved relative to this script: <repo>/third_party/unitree_legged_sdk/
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SDK_ROOT = _REPO_ROOT / "third_party" / "unitree_legged_sdk"

def _find_sdk_lib() -> Path:
    for arch in ("amd64", "arm64"):
        candidate = _SDK_ROOT / "lib" / "python" / arch
        if any(candidate.glob("robot_interface*.so")):
            return candidate
    raise FileNotFoundError(
        f"robot_interface.so not found under {_SDK_ROOT}/lib/python/.\n"
        "Build the SDK: cd third_party/unitree_legged_sdk/build && "
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

JOINT_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]

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

# UDP/state diagnostics
WARMUP_CYCLES = 20
STALE_STATE_WARN_EVERY = 50
STALE_STATE_WARN_AFTER = 5

# Transition between standup and walking. Keep streaming commands during this
# window so low-level control does not drop while waiting for the next phase.
POST_STANDUP_HOLD_S = 1.0
POLICY_POSE_BLEND_S = 1.0
EVAL_PRINT_EVERY_STEPS = 50
MEASURE_PRINT_EVERY_STEPS = 25
MEASURE_KP = 8.0
MEASURE_KD = 0.8
MEASURE_MAX_DELTA = 0.15
MEASURE_TARGET_ALPHA = 0.2
JOINT_TEST_KP = 8.0
JOINT_TEST_KD = 0.8
JOINT_TEST_DELTA = 0.05
JOINT_TEST_DURATION_S = 2.0

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

# Real-robot stand pose and hip gravity-compensation used during standup/stand-only.
# These values are closer to Unitree's low-level examples than the policy pose above.
STAND_DOF_POS = np.array(
    [
        0.0,  1.2, -2.0,  # FR_hip, FR_thigh, FR_calf
        0.0,  1.2, -2.0,  # FL_hip, FL_thigh, FL_calf
        0.0,  1.2, -2.0,  # RR_hip, RR_thigh, RR_calf
        0.0,  1.2, -2.0,  # RL_hip, RL_thigh, RL_calf
    ],
    dtype=np.float32,
)
STAND_HIP_COMP_TAU = np.array(
    [
        -0.65, 0.0, 0.0,  # FR
        +0.65, 0.0, 0.0,  # FL
        -0.65, 0.0, 0.0,  # RR
        +0.65, 0.0, 0.0,  # RL
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
    cfg = None

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
    feedforward_tau: np.ndarray | None = None,
) -> None:
    """Fill LowCmd with position targets and PD gains for all motors."""
    for i in range(NUM_MOTORS):
        cmd.motorCmd[i].mode = 0x0A  # FOC / servo mode
        cmd.motorCmd[i].q    = float(target_q[i])
        cmd.motorCmd[i].dq   = 0.0
        cmd.motorCmd[i].tau  = 0.0 if feedforward_tau is None else float(feedforward_tau[i])
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


def _exchange_udp(udp: sdk.UDP, cmd: sdk.LowCmd, state: sdk.LowState) -> int:
    """Send the current command once and fetch the latest LowState."""
    udp.SetSend(cmd)
    udp.Send()
    udp.Recv()
    udp.GetRecv(state)
    return int(state.tick)


def _recv_state(udp: sdk.UDP, state: sdk.LowState) -> int:
    """Fetch one LowState sample without sending a new command."""
    udp.Recv()
    udp.GetRecv(state)
    return int(state.tick)


def _read_robot_state(state: sdk.LowState) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract arrays from LowState in deploy joint order."""
    quat = np.array(state.imu.quaternion, dtype=np.float32)  # [w, x, y, z]
    gyro = np.array(state.imu.gyroscope, dtype=np.float32)
    dof_pos = np.array([state.motorState[i].q for i in range(NUM_MOTORS)], dtype=np.float32)
    dof_vel = np.array([state.motorState[i].dq for i in range(NUM_MOTORS)], dtype=np.float32)
    tau_est = np.array([state.motorState[i].tauEst for i in range(NUM_MOTORS)], dtype=np.float32)
    return quat, gyro, dof_pos, dof_vel, tau_est


def _build_obs(commands: np.ndarray, quat: np.ndarray, gyro: np.ndarray, dof_pos: np.ndarray, dof_vel: np.ndarray,
               actions: np.ndarray) -> np.ndarray:
    """Build the 45-D observation expected by the policy."""
    pg = _projected_gravity(quat)
    return np.concatenate([
        gyro * _OBS_SCALE_ANG_VEL,
        pg,
        commands * _COMMANDS_SCALE,
        (dof_pos - DEFAULT_DOF_POS) * _OBS_SCALE_DOF_POS,
        dof_vel * _OBS_SCALE_DOF_VEL,
        actions,
    ])


def _build_stand_dof_pos(args: argparse.Namespace) -> np.ndarray:
    return np.array(
        [
            args.stand_front_hip,
            args.stand_front_thigh,
            args.stand_front_calf,
            args.stand_front_hip,
            args.stand_front_thigh,
            args.stand_front_calf,
            args.stand_rear_hip,
            args.stand_rear_thigh,
            args.stand_rear_calf,
            args.stand_rear_hip,
            args.stand_rear_thigh,
            args.stand_rear_calf,
        ],
        dtype=np.float32,
    )


def _build_hip_comp_torque(magnitude: float) -> np.ndarray:
    magnitude = abs(magnitude)
    return np.array(
        [
            -magnitude, 0.0, 0.0,  # FR
            +magnitude, 0.0, 0.0,  # FL
            -magnitude, 0.0, 0.0,  # RR
            +magnitude, 0.0, 0.0,  # RL
        ],
        dtype=np.float32,
    )


def _clip_target_around_anchor(anchor_q: np.ndarray, requested_q: np.ndarray, max_delta: float) -> np.ndarray:
    """Limit commanded joint targets to a small band around an anchor pose."""
    if max_delta <= 0.0:
        return requested_q.copy()
    return anchor_q + np.clip(requested_q - anchor_q, -max_delta, max_delta)


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
    parser.add_argument(
        "--stand_only",
        action="store_true",
        help="Stand up and hold the real-robot stand pose. Do not start the walking policy.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Do not move the robot. Read live state, run policy inference, and log/print model outputs only.",
    )
    parser.add_argument(
        "--measure_only",
        action="store_true",
        help="Apply small clipped policy-driven motions around the current pose and log command/response pairs.",
    )
    parser.add_argument(
        "--joint_test",
        action="store_true",
        help="Move one joint by a small offset from the current pose to verify LowCmd acceptance.",
    )
    parser.add_argument(
        "--eval_duration_s",
        type=float,
        default=0.0,
        help="Optional evaluation duration in seconds for --eval_only. 0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--measure_duration_s",
        type=float,
        default=0.0,
        help="Optional duration in seconds for --measure_only. 0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--measure_max_delta",
        type=float,
        default=MEASURE_MAX_DELTA,
        help="Maximum per-joint deviation [rad] from the initial pose during --measure_only.",
    )
    parser.add_argument(
        "--measure_alpha",
        type=float,
        default=MEASURE_TARGET_ALPHA,
        help="Low-pass factor for commanded joint targets during --measure_only. Range: 0-1.",
    )
    parser.add_argument("--measure_kp", type=float, default=MEASURE_KP, help="Position gain for --measure_only.")
    parser.add_argument("--measure_kd", type=float, default=MEASURE_KD, help="Velocity gain for --measure_only.")
    parser.add_argument(
        "--joint_index",
        type=int,
        default=7,
        help="Joint index for --joint_test. 0-11 follow the FR/FL/RR/RL order shown in the header.",
    )
    parser.add_argument(
        "--joint_delta",
        type=float,
        default=JOINT_TEST_DELTA,
        help="Target offset [rad] added to the current joint angle during --joint_test.",
    )
    parser.add_argument(
        "--joint_duration_s",
        type=float,
        default=JOINT_TEST_DURATION_S,
        help="How long to hold the offset target during --joint_test.",
    )
    parser.add_argument("--joint_kp", type=float, default=JOINT_TEST_KP, help="Position gain for --joint_test.")
    parser.add_argument("--joint_kd", type=float, default=JOINT_TEST_KD, help="Velocity gain for --joint_test.")
    parser.add_argument("--stand_front_hip", type=float, default=float(STAND_DOF_POS[0]))
    parser.add_argument("--stand_front_thigh", type=float, default=float(STAND_DOF_POS[1]))
    parser.add_argument("--stand_front_calf", type=float, default=float(STAND_DOF_POS[2]))
    parser.add_argument("--stand_rear_hip", type=float, default=float(STAND_DOF_POS[6]))
    parser.add_argument("--stand_rear_thigh", type=float, default=float(STAND_DOF_POS[7]))
    parser.add_argument("--stand_rear_calf", type=float, default=float(STAND_DOF_POS[8]))
    parser.add_argument(
        "--stand_hip_comp",
        type=float,
        default=0.65,
        help="Feedforward hip torque magnitude [Nm] used during standup/stand-only. Set 0 to disable.",
    )
    parser.add_argument("--log", action="store_true",       help="Enable data logging to CSV")
    parser.add_argument("--log_dir", type=str, default="deploy_logs",
                        help="Directory for CSV log output")
    args = parser.parse_args()
    active_mode_count = sum([args.stand_only, args.eval_only, args.measure_only, args.joint_test])
    if active_mode_count > 1:
        raise ValueError("--stand_only, --eval_only, --measure_only, and --joint_test are mutually exclusive.")
    args.measure_alpha = float(np.clip(args.measure_alpha, 0.0, 1.0))
    if not 0 <= args.joint_index < NUM_MOTORS:
        raise ValueError(f"--joint_index must be in [0, {NUM_MOTORS - 1}].")

    # ------------------------------------------------------------------
    # Load configs and policy
    # ------------------------------------------------------------------
    policy = None
    if not args.joint_test:
        cfgs_path = os.path.join(args.model_dir, "cfgs.pkl")
        with open(cfgs_path, "rb") as f:
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

        print(f"[deploy] Loading policy from {args.model_dir}/model_{args.ckpt}.pt ...")
        policy = load_policy(args.model_dir, args.ckpt, train_cfg)

    commands = np.array([args.vx, args.vy, args.wz], dtype=np.float32)
    stand_dof_pos = _build_stand_dof_pos(args)
    stand_hip_comp_tau = _build_hip_comp_torque(args.stand_hip_comp)
    print(f"[deploy] Commands: vx={args.vx} m/s  vy={args.vy} m/s  wz={args.wz} rad/s")
    print(f"[deploy] Stand q  : {np.round(stand_dof_pos, 3)}")
    print(f"[deploy] Hip ff   : {np.round(stand_hip_comp_tau[[0, 3, 6, 9]], 3)}")

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
    LOWLEVEL = 0xff
    safe = sdk.Safety(sdk.LeggedType.Go1)
    udp  = sdk.UDP(LOWLEVEL, LOCAL_PORT, ROBOT_IP, ROBOT_PORT)
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
    _set_damping(cmd)
    warmup_ticks: list[int] = []
    for _ in range(WARMUP_CYCLES):
        warmup_ticks.append(_exchange_udp(udp, cmd, state))
        time.sleep(0.002)

    q_init = np.array([state.motorState[i].q for i in range(NUM_MOTORS)], dtype=np.float32)
    unique_ticks = len(set(warmup_ticks))
    if unique_ticks < 2:
        raise RuntimeError(
            "[deploy] LowState tick did not advance during warm-up. "
            "UDP link is likely not exchanging fresh packets."
        )
    if np.allclose(q_init, 0.0, atol=1e-3):
        raise RuntimeError(
            "[deploy] Initial joint positions are all near zero after warm-up. "
            "Refusing to continue because robot feedback looks invalid."
        )
    print(f"[deploy] Initial tick: {warmup_ticks[-1]}  (unique ticks during warm-up: {unique_ticks})")
    print(f"[deploy] Initial q : {np.round(q_init, 3)}")
    print(f"[deploy] Policy q  : {np.round(DEFAULT_DOF_POS, 3)}")

    if args.joint_test:
        step_count = 0
        overrun_count = 0
        last_state_tick = int(state.tick)
        stale_state_count = 0
        actions = np.zeros(NUM_MOTORS, dtype=np.float32)
        target_q = q_init.copy()
        target_q[args.joint_index] += args.joint_delta
        joint_name = JOINT_NAMES[args.joint_index]

        print(
            f"[deploy] Joint-test target: {joint_name} (index {args.joint_index})  "
            f"delta={args.joint_delta:.3f} rad  hold={args.joint_duration_s:.2f} s"
        )
        print(
            f"[deploy] Joint-test q0={q_init[args.joint_index]:.3f}  "
            f"q_target={target_q[args.joint_index]:.3f}  "
            f"kp={args.joint_kp:.1f}  kd={args.joint_kd:.1f}"
        )
        input(
            "\n[deploy] *** Verify the robot is supported and safe for a single-joint motion. ***\n"
            "         Press Enter to begin JOINT TEST ..."
        )

        test_t0 = time.monotonic()
        print("[deploy] Joint-test mode active. Press Ctrl+C to stop.")

        while running:
            loop_t0 = time.monotonic()
            elapsed = loop_t0 - test_t0
            if elapsed >= args.joint_duration_s:
                print(f"[deploy] Joint-test duration reached ({args.joint_duration_s:.2f} s).")
                break

            current_tick = _recv_state(udp, state)
            retry_count = 0
            while current_tick == last_state_tick and retry_count < 3:
                time.sleep(0.001)
                current_tick = _recv_state(udp, state)
                retry_count += 1
            if current_tick == last_state_tick:
                stale_state_count += 1
                if stale_state_count >= STALE_STATE_WARN_AFTER and stale_state_count % STALE_STATE_WARN_EVERY == 0:
                    print(
                        f"[deploy] Warning: stale LowState tick={current_tick} "
                        f"for {stale_state_count} consecutive cycle(s)."
                    )
            else:
                stale_state_count = 0
            last_state_tick = current_tick

            quat, gyro, dof_pos, dof_vel, tau_est = _read_robot_state(state)
            roll_deg, pitch_deg, _ = _euler_from_quat_deg(quat)
            if abs(roll_deg) > EMERGENCY_ROLL_DEG or abs(pitch_deg) > EMERGENCY_PITCH_DEG:
                print(
                    f"[deploy] EMERGENCY STOP: roll={roll_deg:.1f}deg pitch={pitch_deg:.1f}deg "
                    f"exceeds limit ({EMERGENCY_ROLL_DEG}deg/{EMERGENCY_PITCH_DEG}deg)"
                )
                running = False
                break

            _set_motor_cmd(cmd, target_q, args.joint_kp, args.joint_kd)
            safe.PowerProtect(cmd, state, 6)
            udp.SetSend(cmd)
            udp.Send()

            if logger is not None:
                logger.log(
                    timestamp=time.monotonic(),
                    state_tick=current_tick,
                    dof_pos=dof_pos,
                    dof_vel=dof_vel,
                    tau_est=tau_est,
                    imu_gyro=gyro,
                    imu_quat=quat,
                    commands=commands,
                    actions=actions,
                    sent_target_dof_pos=target_q,
                )

            if step_count % EVAL_PRINT_EVERY_STEPS == 0:
                measured_delta = dof_pos[args.joint_index] - q_init[args.joint_index]
                target_error = target_q[args.joint_index] - dof_pos[args.joint_index]
                print(
                    "[deploy] Joint-test "
                    f"step={step_count} tick={current_tick} "
                    f"joint={joint_name} "
                    f"q_cur={dof_pos[args.joint_index]:.3f} "
                    f"q_target={target_q[args.joint_index]:.3f} "
                    f"delta_measured={measured_delta:.3f} "
                    f"target_error={target_error:.3f}"
                )

            step_count += 1

            dt_used = time.monotonic() - loop_t0
            remaining = CONTROL_DT - dt_used
            if remaining > 0:
                time.sleep(remaining)
            else:
                overrun_count += 1
                if overrun_count % 50 == 1:
                    print(f"[deploy] Loop overrun: {-remaining * 1000:.1f} ms late  (step {step_count})")

        q_end = np.array([state.motorState[i].q for i in range(NUM_MOTORS)], dtype=np.float32)
        delta_measured = q_end[args.joint_index] - q_init[args.joint_index]
        print(
            f"[deploy] Joint-test result: q_end={q_end[args.joint_index]:.3f}  "
            f"delta_measured={delta_measured:.3f} rad"
        )
        print(f"[deploy] Joint-test stopping after {step_count} steps - entering damping mode ...")
        _set_damping(cmd)
        udp.SetSend(cmd)
        udp.Send()
        time.sleep(0.5)
        if logger is not None:
            logger.close()
        print("[deploy] Done.")
        return

    if args.eval_only:
        actions = np.zeros(NUM_MOTORS, dtype=np.float32)
        step_count = 0
        overrun_count = 0
        last_state_tick = int(state.tick)
        stale_state_count = 0
        eval_t0 = time.monotonic()

        print("[deploy] Eval-only mode active. Robot stays in damping. Press Ctrl+C to stop.")

        while running:
            loop_t0 = time.monotonic()
            if args.eval_duration_s > 0.0 and (loop_t0 - eval_t0) >= args.eval_duration_s:
                print(f"[deploy] Eval-only duration reached ({args.eval_duration_s:.1f} s).")
                break

            current_tick = _recv_state(udp, state)
            retry_count = 0
            while current_tick == last_state_tick and retry_count < 3:
                time.sleep(0.001)
                current_tick = _recv_state(udp, state)
                retry_count += 1
            if current_tick == last_state_tick:
                stale_state_count += 1
                if stale_state_count >= STALE_STATE_WARN_AFTER and stale_state_count % STALE_STATE_WARN_EVERY == 0:
                    print(
                        f"[deploy] Warning: stale LowState tick={current_tick} "
                        f"for {stale_state_count} consecutive cycle(s)."
                    )
            else:
                stale_state_count = 0
            last_state_tick = current_tick

            quat, gyro, dof_pos, dof_vel, tau_est = _read_robot_state(state)
            roll_deg, pitch_deg, _ = _euler_from_quat_deg(quat)
            obs_np = _build_obs(commands, quat, gyro, dof_pos, dof_vel, actions)
            obs_tensor = torch.from_numpy(obs_np).unsqueeze(0)
            obs_dict = TensorDict({"policy": obs_tensor}, batch_size=[1])

            with torch.no_grad():
                new_actions_tensor = policy(obs_dict)

            new_actions = new_actions_tensor.cpu().numpy().squeeze(0).astype(np.float32)
            new_actions = np.clip(new_actions, -CLIP_ACTIONS, CLIP_ACTIONS)
            target_dof_pos = new_actions * ACTION_SCALE + DEFAULT_DOF_POS

            if logger is not None:
                logger.log(
                    timestamp=time.monotonic(),
                    state_tick=current_tick,
                    dof_pos=dof_pos,
                    dof_vel=dof_vel,
                    tau_est=tau_est,
                    imu_gyro=gyro,
                    imu_quat=quat,
                    commands=commands,
                    actions=new_actions,
                    policy_target_dof_pos=target_dof_pos,
                )

            if step_count % EVAL_PRINT_EVERY_STEPS == 0:
                target_err = target_dof_pos - dof_pos
                joint_delta = dof_pos - q_init
                print(
                    "[deploy] Eval "
                    f"step={step_count} tick={current_tick} "
                    f"roll={roll_deg:.1f} pitch={pitch_deg:.1f} "
                    f"|action|_max={np.max(np.abs(new_actions)):.3f} "
                    f"target_err_mean={np.mean(np.abs(target_err)):.3f} "
                    f"target_err_max={np.max(np.abs(target_err)):.3f}"
                    f" max_joint_delta={np.max(np.abs(joint_delta)):.3f}"
                )
                print(
                    f"         FR q_cur={np.round(dof_pos[0:3], 3)}  "
                    f"q_tgt={np.round(target_dof_pos[0:3], 3)}"
                )
                print(
                    f"         FL q_cur={np.round(dof_pos[3:6], 3)}  "
                    f"q_tgt={np.round(target_dof_pos[3:6], 3)}"
                )
                print(
                    f"         RR q_cur={np.round(dof_pos[6:9], 3)}  "
                    f"q_tgt={np.round(target_dof_pos[6:9], 3)}"
                )
                print(
                    f"         RL q_cur={np.round(dof_pos[9:12], 3)}  "
                    f"q_tgt={np.round(target_dof_pos[9:12], 3)}"
                )

            _set_damping(cmd)
            udp.SetSend(cmd)
            udp.Send()

            actions = new_actions
            step_count += 1

            dt_used = time.monotonic() - loop_t0
            remaining = CONTROL_DT - dt_used
            if remaining > 0:
                time.sleep(remaining)
            else:
                overrun_count += 1
                if overrun_count % 50 == 1:
                    print(f"[deploy] Loop overrun: {-remaining * 1000:.1f} ms late  (step {step_count})")

        print(f"[deploy] Eval-only stopping after {step_count} steps — entering damping mode ...")
        _set_damping(cmd)
        udp.SetSend(cmd)
        udp.Send()
        time.sleep(0.5)
        if logger is not None:
            logger.close()
        print("[deploy] Done.")
        return

    if args.measure_only:
        actions = np.zeros(NUM_MOTORS, dtype=np.float32)
        step_count = 0
        overrun_count = 0
        last_state_tick = int(state.tick)
        stale_state_count = 0
        anchor_q = q_init.copy()
        sent_target_q = anchor_q.copy()

        print(f"[deploy] Measure anchor q : {np.round(anchor_q, 3)}")
        print(
            "[deploy] Measure params : "
            f"max_delta={args.measure_max_delta:.3f} rad  "
            f"alpha={args.measure_alpha:.2f}  "
            f"kp={args.measure_kp:.1f}  kd={args.measure_kd:.1f}"
        )
        input(
            "\n[deploy] *** Verify the robot is supported and safe for small joint motions. ***\n"
            "         Press Enter to begin MEASUREMENT mode ..."
        )
        measure_t0 = time.monotonic()
        print("[deploy] Measurement mode active. Press Ctrl+C to stop.")

        while running:
            loop_t0 = time.monotonic()
            if args.measure_duration_s > 0.0 and (loop_t0 - measure_t0) >= args.measure_duration_s:
                print(f"[deploy] Measurement duration reached ({args.measure_duration_s:.1f} s).")
                break

            current_tick = _recv_state(udp, state)
            retry_count = 0
            while current_tick == last_state_tick and retry_count < 3:
                time.sleep(0.001)
                current_tick = _recv_state(udp, state)
                retry_count += 1
            if current_tick == last_state_tick:
                stale_state_count += 1
                if stale_state_count >= STALE_STATE_WARN_AFTER and stale_state_count % STALE_STATE_WARN_EVERY == 0:
                    print(
                        f"[deploy] Warning: stale LowState tick={current_tick} "
                        f"for {stale_state_count} consecutive cycle(s)."
                    )
            else:
                stale_state_count = 0
            last_state_tick = current_tick

            quat, gyro, dof_pos, dof_vel, tau_est = _read_robot_state(state)
            roll_deg, pitch_deg, _ = _euler_from_quat_deg(quat)
            if abs(roll_deg) > EMERGENCY_ROLL_DEG or abs(pitch_deg) > EMERGENCY_PITCH_DEG:
                print(
                    f"[deploy] EMERGENCY STOP: roll={roll_deg:.1f}deg pitch={pitch_deg:.1f}deg "
                    f"exceeds limit ({EMERGENCY_ROLL_DEG}deg/{EMERGENCY_PITCH_DEG}deg)"
                )
                running = False
                break

            obs_np = _build_obs(commands, quat, gyro, dof_pos, dof_vel, actions)
            obs_tensor = torch.from_numpy(obs_np).unsqueeze(0)
            obs_dict = TensorDict({"policy": obs_tensor}, batch_size=[1])

            with torch.no_grad():
                new_actions_tensor = policy(obs_dict)

            new_actions = new_actions_tensor.cpu().numpy().squeeze(0).astype(np.float32)
            new_actions = np.clip(new_actions, -CLIP_ACTIONS, CLIP_ACTIONS)
            policy_target_q = new_actions * ACTION_SCALE + DEFAULT_DOF_POS
            clipped_target_q = _clip_target_around_anchor(anchor_q, policy_target_q, args.measure_max_delta)
            sent_target_q = sent_target_q * (1.0 - args.measure_alpha) + clipped_target_q * args.measure_alpha

            _set_motor_cmd(cmd, sent_target_q, args.measure_kp, args.measure_kd)
            safe.PowerProtect(cmd, state, 6)
            udp.SetSend(cmd)
            udp.Send()

            if logger is not None:
                logger.log(
                    timestamp=time.monotonic(),
                    state_tick=current_tick,
                    dof_pos=dof_pos,
                    dof_vel=dof_vel,
                    tau_est=tau_est,
                    imu_gyro=gyro,
                    imu_quat=quat,
                    commands=commands,
                    actions=new_actions,
                    policy_target_dof_pos=policy_target_q,
                    sent_target_dof_pos=sent_target_q,
                )

            if step_count % MEASURE_PRINT_EVERY_STEPS == 0:
                policy_err = policy_target_q - dof_pos
                sent_err = sent_target_q - dof_pos
                print(
                    "[deploy] Measure "
                    f"step={step_count} tick={current_tick} "
                    f"roll={roll_deg:.1f} pitch={pitch_deg:.1f} "
                    f"|action|_max={np.max(np.abs(new_actions)):.3f} "
                    f"policy_err_max={np.max(np.abs(policy_err)):.3f} "
                    f"sent_err_max={np.max(np.abs(sent_err)):.3f}"
                )
                print(
                    f"         q_cur={np.round(dof_pos[:3], 3)}  "
                    f"q_pol={np.round(policy_target_q[:3], 3)}  "
                    f"q_cmd={np.round(sent_target_q[:3], 3)}"
                )

            actions = new_actions
            step_count += 1

            dt_used = time.monotonic() - loop_t0
            remaining = CONTROL_DT - dt_used
            if remaining > 0:
                time.sleep(remaining)
            else:
                overrun_count += 1
                if overrun_count % 50 == 1:
                    print(f"[deploy] Loop overrun: {-remaining * 1000:.1f} ms late  (step {step_count})")

        print(f"[deploy] Measurement stopping after {step_count} steps - entering damping mode ...")
        _set_damping(cmd)
        udp.SetSend(cmd)
        udp.Send()
        time.sleep(0.5)
        if logger is not None:
            logger.close()
        print("[deploy] Done.")
        return

    # ------------------------------------------------------------------
    # Standup phase: interpolate from q_init to stand_dof_pos over 2 s
    # Uses low Kp/Kd for safe, slow movement.
    # ------------------------------------------------------------------
    input(
        "\n[deploy] *** Verify the robot is suspended or on flat ground. ***\n"
        "         Press Enter to begin STANDUP sequence ..."
    )

    standup_steps = int(STANDUP_DURATION_S / CONTROL_DT)
    print(f"[deploy] Standing up ({STANDUP_DURATION_S:.1f} s, {standup_steps} steps) ...")

    for step_i in range(standup_steps):
        if not running:
            break
        t0 = time.monotonic()
        rate = (step_i + 1) / standup_steps
        target_q = q_init * (1.0 - rate) + stand_dof_pos * rate

        _set_motor_cmd(cmd, target_q, STANDUP_KP, STANDUP_KD, feedforward_tau=stand_hip_comp_tau)
        safe.PowerProtect(cmd, state, 6)
        _exchange_udp(udp, cmd, state)

        dt_used = time.monotonic() - t0
        remaining = CONTROL_DT - dt_used
        if remaining > 0:
            time.sleep(remaining)

    if not running:
        print("[deploy] Standup interrupted before walking.")
    else:
        hold_steps = max(1, int(POST_STANDUP_HOLD_S / CONTROL_DT))
        print(
            f"[deploy] Standup complete. Holding stand pose for "
            f"{POST_STANDUP_HOLD_S:.1f} s before WALKING ..."
        )
        for _ in range(hold_steps):
            if not running:
                break
            t0 = time.monotonic()
            _set_motor_cmd(cmd, stand_dof_pos, STANDUP_KP, STANDUP_KD, feedforward_tau=stand_hip_comp_tau)
            safe.PowerProtect(cmd, state, 6)
            _exchange_udp(udp, cmd, state)

            dt_used = time.monotonic() - t0
            remaining = CONTROL_DT - dt_used
            if remaining > 0:
                time.sleep(remaining)

        q_ready = np.array([state.motorState[i].q for i in range(NUM_MOTORS)], dtype=np.float32)
        pose_err = q_ready - stand_dof_pos
        print(f"[deploy] Ready q   : {np.round(q_ready, 3)}")
        print(
            f"[deploy] Pose error: mean={np.mean(np.abs(pose_err)):.3f} rad  "
            f"max={np.max(np.abs(pose_err)):.3f} rad"
        )

        if args.stand_only and running:
            print("[deploy] Stand-only mode active. Holding stand pose. Press Ctrl+C to stop.")
            while running:
                t0 = time.monotonic()
                _set_motor_cmd(cmd, stand_dof_pos, STANDUP_KP, STANDUP_KD, feedforward_tau=stand_hip_comp_tau)
                safe.PowerProtect(cmd, state, 6)
                _exchange_udp(udp, cmd, state)

                dt_used = time.monotonic() - t0
                remaining = CONTROL_DT - dt_used
                if remaining > 0:
                    time.sleep(remaining)

        elif running:
            blend_steps = max(1, int(POLICY_POSE_BLEND_S / CONTROL_DT))
            print(
                f"[deploy] Blending stand pose to policy pose for "
                f"{POLICY_POSE_BLEND_S:.1f} s before WALKING ..."
            )
            for step_i in range(blend_steps):
                if not running:
                    break
                t0 = time.monotonic()
                rate = (step_i + 1) / blend_steps
                target_q = stand_dof_pos * (1.0 - rate) + DEFAULT_DOF_POS * rate
                ff_tau = stand_hip_comp_tau * (1.0 - rate)
                _set_motor_cmd(cmd, target_q, STANDUP_KP, STANDUP_KD, feedforward_tau=ff_tau)
                safe.PowerProtect(cmd, state, 6)
                _exchange_udp(udp, cmd, state)

                dt_used = time.monotonic() - t0
                remaining = CONTROL_DT - dt_used
                if remaining > 0:
                    time.sleep(remaining)

    # ------------------------------------------------------------------
    # Main control loop  (50 Hz)
    #
    # Replicates go1_env.step() + _update_observation() exactly:
    #   - obs uses the previous policy output
    #   - motor receives that same previous policy output
    #   - policy computes the next action from the current observation
    # ------------------------------------------------------------------
    actions = np.zeros(NUM_MOTORS, dtype=np.float32)  # previous policy output; used in obs and sent this step
    step_count = 0
    overrun_count = 0
    last_state_tick = int(state.tick)
    stale_state_count = 0

    if running:
        print("[deploy] Walking policy active. Press Ctrl+C to stop.")

    while running:
        t0 = time.monotonic()

        # 1. Receive robot state
        current_tick = _recv_state(udp, state)
        retry_count = 0
        while current_tick == last_state_tick and retry_count < 3:
            time.sleep(0.001)
            current_tick = _recv_state(udp, state)
            retry_count += 1
        if current_tick == last_state_tick:
            stale_state_count += 1
            if stale_state_count >= STALE_STATE_WARN_AFTER and stale_state_count % STALE_STATE_WARN_EVERY == 0:
                print(
                    f"[deploy] Warning: stale LowState tick={current_tick} "
                    f"for {stale_state_count} consecutive cycle(s)."
                )
        else:
            stale_state_count = 0
        last_state_tick = current_tick

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

        # 5. Compute motor target using the previous policy output
        #    (replicates simulate_action_latency=True from training)
        exec_actions  = actions
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
                state_tick=current_tick,
                dof_pos=dof_pos,
                dof_vel=dof_vel,
                tau_est=tau_est,
                imu_gyro=gyro,
                imu_quat=quat,
                commands=commands,
                actions=new_actions,
                policy_target_dof_pos=new_actions * ACTION_SCALE + DEFAULT_DOF_POS,
                sent_target_dof_pos=target_dof_pos,
            )

        # 8. Advance buffers
        actions = new_actions
        step_count += 1

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

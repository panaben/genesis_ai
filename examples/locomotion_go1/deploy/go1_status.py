#!/usr/bin/env python3
"""
Read LowState from a Unitree Go1 and print IMU + per-motor status.

By default this script sends a passive low-level damping packet before each
receive so that we can get fresh LowState without re-asserting an arbitrary
servo command. `--listen-only` can be used if you explicitly want passive
receive behavior, and `--send-sdk-default` can be used to reproduce the raw
SDK communication test behavior.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SDK_ROOT = _REPO_ROOT / "third_party" / "unitree_legged_sdk"


def _find_sdk_lib() -> Path:
    for arch in ("amd64", "arm64"):
        candidate = _SDK_ROOT / "lib" / "python" / arch
        if any(candidate.glob("robot_interface*.so")) or any(candidate.glob("robot_interface*.pyd")):
            return candidate
    raise FileNotFoundError(
        f"robot_interface library not found under {_SDK_ROOT / 'lib' / 'python'}.\n"
        "Build the SDK Python wrapper first."
    )


sys.path.insert(0, str(_find_sdk_lib()))
import robot_interface as sdk  # noqa: E402


LOWLEVEL = 0xFF
ROBOT_IP = "192.168.123.10"
ROBOT_PORT = 8007
LOCAL_PORT = 8090
JOINT_NAMES = [
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
]


def _mode_name(mode: int) -> str:
    if mode == 0x00:
        return "damping"
    if mode == 0x0A:
        return "servo"
    if mode == 0x08:
        return "overheat"
    return f"0x{mode:02X}"


def _euler_from_quat_deg(q_wxyz: list[float] | tuple[float, ...]) -> tuple[float, float, float]:
    w, x, y, z = [float(v) for v in q_wxyz]
    roll = math.degrees(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))
    yaw = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return roll, pitch, yaw


def _fill_damping_cmd(cmd: sdk.LowCmd) -> None:
    for i in range(len(JOINT_NAMES)):
        cmd.motorCmd[i].mode = 0x00
        cmd.motorCmd[i].q = 0.0
        cmd.motorCmd[i].dq = 0.0
        cmd.motorCmd[i].tau = 0.0
        cmd.motorCmd[i].Kp = 0.0
        cmd.motorCmd[i].Kd = 0.0


def _print_state(state: sdk.LowState) -> None:
    roll_deg, pitch_deg, yaw_deg = _euler_from_quat_deg(state.imu.quaternion)
    print(
        f"tick={int(state.tick)}  "
        f"rpy_deg=({roll_deg:+6.1f}, {pitch_deg:+6.1f}, {yaw_deg:+6.1f})  "
        f"gyro=({state.imu.gyroscope[0]:+6.2f}, {state.imu.gyroscope[1]:+6.2f}, {state.imu.gyroscope[2]:+6.2f})"
    )
    for i, joint_name in enumerate(JOINT_NAMES):
        motor = state.motorState[i]
        print(
            f"  {i:02d} {joint_name:<8} "
            f"mode={_mode_name(int(motor.mode)):<8} "
            f"q={float(motor.q):+7.3f} "
            f"dq={float(motor.dq):+7.3f} "
            f"tau={float(motor.tauEst):+7.3f} "
            f"temp={int(motor.temperature):3d}C"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Read Go1 LowState for diagnosis.")
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--robot-port", type=int, default=ROBOT_PORT)
    parser.add_argument("--local-port", type=int, default=LOCAL_PORT)
    parser.add_argument(
        "--cycles",
        type=int,
        default=5,
        help="How many samples to print. Use 0 to keep printing until Ctrl+C.",
    )
    parser.add_argument("--interval", type=float, default=0.2, help="Delay between printed samples in seconds.")
    parser.add_argument("--listen-only", action="store_true", help="Do not send any packet before receiving.")
    parser.add_argument(
        "--send-sdk-default",
        action="store_true",
        help="Send the SDK-initialized LowCmd instead of passive damping.",
    )
    args = parser.parse_args()

    udp = sdk.UDP(LOWLEVEL, args.local_port, args.robot_ip, args.robot_port)
    state = sdk.LowState()
    cmd = sdk.LowCmd()
    udp.InitCmdData(cmd)
    if args.listen_only:
        print("[status] Reading LowState only. No command is sent to the robot.")
    elif args.send_sdk_default:
        print("[status] Using send/recv low-level exchange with SDK-initialized LowCmd.")
    else:
        _fill_damping_cmd(cmd)
        print("[status] Using send/recv low-level exchange with passive damping packets.")
    print("[status] If tick never changes, the robot may not be replying on the low-level endpoint.")

    sample_count = 0
    last_tick: int | None = None
    stale_count = 0

    try:
        while args.cycles == 0 or sample_count < args.cycles:
            if not args.listen_only:
                udp.SetSend(cmd)
                udp.Send()
            udp.Recv()
            udp.GetRecv(state)
            tick = int(state.tick)
            if last_tick is not None and tick == last_tick:
                stale_count += 1
            last_tick = tick

            print(f"\n[status] sample={sample_count} stale_count={stale_count}")
            _print_state(state)
            sample_count += 1
            time.sleep(args.interval)

        if last_tick == 0 and stale_count >= max(0, sample_count - 1):
            print("\n[status] No fresh LowState was observed.")
            if args.listen_only:
                print("[status] Try again without --listen-only to force a low-level send/recv exchange.")
    except KeyboardInterrupt:
        print("\n[status] Stopped by user.")


if __name__ == "__main__":
    main()

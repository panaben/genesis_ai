#!/usr/bin/env python3
"""
Send a minimal HighLevel recovery / stand-up sequence to a Unitree Go1.

Use this only with the robot supported or with a spotter ready.
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


HIGHLEVEL = 0xEE
ROBOT_IP = "192.168.123.161"
ROBOT_PORT = 8082
LOCAL_PORT = 8090


def _fill_default_cmd(cmd: sdk.HighCmd) -> None:
    cmd.mode = 0
    cmd.gaitType = 0
    cmd.speedLevel = 0
    cmd.footRaiseHeight = 0.0
    cmd.bodyHeight = 0.0
    cmd.position = [0.0, 0.0]
    cmd.euler = [0.0, 0.0, 0.0]
    cmd.velocity = [0.0, 0.0]
    cmd.yawSpeed = 0.0
    cmd.reserve = 0


def _euler_from_quat_deg(q_wxyz: list[float] | tuple[float, ...]) -> tuple[float, float, float]:
    w, x, y, z = [float(v) for v in q_wxyz]
    roll = math.degrees(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))
    yaw = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return roll, pitch, yaw


def _run_phase(
    udp: sdk.UDP,
    cmd: sdk.HighCmd,
    state: sdk.HighState,
    *,
    mode: int,
    duration_s: float,
    rate_hz: float,
    label: str,
) -> None:
    if duration_s <= 0.0:
        return

    dt = 1.0 / rate_hz
    total_steps = max(1, int(round(duration_s * rate_hz)))
    print(f"[recover] Phase: {label}  mode={mode}  duration={duration_s:.1f}s  steps={total_steps}")

    for step in range(total_steps):
        t0 = time.monotonic()
        udp.Recv()
        udp.GetRecv(state)

        _fill_default_cmd(cmd)
        cmd.mode = mode
        udp.SetSend(cmd)
        udp.Send()

        if step % max(1, int(rate_hz // 2)) == 0 or step == total_steps - 1:
            roll_deg, pitch_deg, yaw_deg = _euler_from_quat_deg(state.imu.quaternion)
            print(
                f"[recover] step={step:04d}/{total_steps:04d} "
                f"robot_mode={int(state.mode)} "
                f"progress={float(state.progress):.2f} "
                f"gait={int(state.gaitType)} "
                f"rpy_deg=({roll_deg:+6.1f}, {pitch_deg:+6.1f}, {yaw_deg:+6.1f})"
            )

        remaining = dt - (time.monotonic() - t0)
        if remaining > 0:
            time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description="Send HighLevel recovery/stand-up commands to Go1.")
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--robot-port", type=int, default=ROBOT_PORT)
    parser.add_argument("--local-port", type=int, default=LOCAL_PORT)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument(
        "--sequence",
        choices=("recovery", "standup", "recovery-stand", "idle"),
        default="recovery-stand",
    )
    parser.add_argument("--recovery-s", type=float, default=3.0)
    parser.add_argument("--standup-s", type=float, default=3.0)
    parser.add_argument("--idle-s", type=float, default=1.5)
    parser.add_argument("--yes", action="store_true", help="Skip the safety prompt.")
    args = parser.parse_args()

    print("[recover] This script will SEND HighLevel commands to the robot.")
    print("[recover] Support the robot, clear the area, and keep the remote emergency stop ready.")
    print("[recover] Recommended order: status check -> recovery-stand -> inspect result.")
    if not args.yes:
        answer = input("[recover] Type YES to continue: ").strip()
        if answer != "YES":
            print("[recover] Aborted.")
            return

    udp = sdk.UDP(HIGHLEVEL, args.local_port, args.robot_ip, args.robot_port)
    cmd = sdk.HighCmd()
    state = sdk.HighState()
    udp.InitCmdData(cmd)

    if args.sequence == "recovery":
        phases = [(8, args.recovery_s, "recovery stand")]
    elif args.sequence == "standup":
        phases = [(6, args.standup_s, "position stand up")]
    elif args.sequence == "idle":
        phases = [(0, args.idle_s, "idle/default stand")]
    else:
        phases = [
            (8, args.recovery_s, "recovery stand"),
            (6, args.standup_s, "position stand up"),
            (0, args.idle_s, "idle/default stand"),
        ]

    try:
        for mode, duration_s, label in phases:
            _run_phase(udp, cmd, state, mode=mode, duration_s=duration_s, rate_hz=args.rate_hz, label=label)
    except KeyboardInterrupt:
        print("\n[recover] Interrupted by user.")
    finally:
        _fill_default_cmd(cmd)
        udp.SetSend(cmd)
        udp.Send()
        print("[recover] Final command: mode=0 (idle/default stand).")


if __name__ == "__main__":
    main()

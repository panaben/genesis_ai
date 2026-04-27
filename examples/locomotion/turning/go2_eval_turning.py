import argparse
import os
import pickle
from importlib import metadata

import torch
from genesis.vis.keybindings import Key, KeyAction, Keybind

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env_turning import Go2Env


MODE_COMMANDS = {
    "neutral": (0.0, 0.0, 0.0),
    "forward": (0.5, 0.0, 0.0),
    "backward": (-0.4, 0.0, 0.0),
    "turn_left": (0.2, 0.0, 0.8),
    "turn_right": (0.2, 0.0, -0.8),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    with open(f"logs/{args.exp_name}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}

    # Widen command ranges for manual backward/turn commands during evaluation.
    command_cfg["lin_vel_x_range"] = [-0.8, 0.8]
    command_cfg["lin_vel_y_range"] = [-0.5, 0.5]
    command_cfg["ang_vel_range"] = [-1.2, 1.2]

    state = {
        "running": True,
        "restart_requested": False,
        "env": None,
        "follow_enabled": True,
        "command_mode": "forward",
    }

    def apply_follow_mode():
        env = state["env"]
        if env is None:
            return
        viewer = env.scene.viewer
        if state["follow_enabled"]:
            viewer.follow_entity(env.robot, fixed_axis=(None, None, None), smoothing=0.15, fix_orientation=False)
            gs.logger.info("Camera follow enabled")
        else:
            # Viewer currently has no public unfollow API; clear follow state explicitly.
            viewer._followed_entity = None
            viewer._follow_fixed_axis = None
            viewer._follow_smoothing = None
            viewer._follow_fix_orientation = None
            viewer._follow_lookat = None
            gs.logger.info("Camera follow disabled")

    def apply_command_mode():
        env = state["env"]
        if env is None:
            return
        command = MODE_COMMANDS[state["command_mode"]]
        env.set_command_override(command)
        gs.logger.info(f"Command mode: {state['command_mode']} command={command}")

    def set_command_mode(mode_name: str):
        state["command_mode"] = mode_name
        apply_command_mode()

    def toggle_motion_pause():
        env = state["env"]
        if env is not None:
            next_paused = not env.is_motion_paused
            env.set_motion_paused(next_paused)
            gs.logger.info("Motion paused" if next_paused else "Motion resumed")

    def restart_simulation():
        state["restart_requested"] = True
        gs.logger.info("Restart requested")

    def stop_program():
        state["running"] = False
        gs.logger.info("Exiting evaluation")

    def toggle_camera_follow():
        state["follow_enabled"] = not state["follow_enabled"]
        apply_follow_mode()

    with torch.no_grad():
        while state["running"]:
            env = Go2Env(
                num_envs=1,
                env_cfg=env_cfg,
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg,
                command_cfg=command_cfg,
                show_viewer=True,
            )
            state["env"] = env
            state["restart_requested"] = False

            apply_follow_mode()
            apply_command_mode()

            env.scene.viewer.register_keybinds(
                Keybind("toggle_pause_motion", Key.F5, KeyAction.RELEASE, callback=toggle_motion_pause),
                Keybind("restart_sim", Key.F6, KeyAction.RELEASE, callback=restart_simulation),
                Keybind("toggle_camera_follow", Key.F7, KeyAction.RELEASE, callback=toggle_camera_follow),
                Keybind("mode_forward", Key.F8, KeyAction.RELEASE, callback=lambda: set_command_mode("forward")),
                Keybind("mode_backward", Key.F9, KeyAction.RELEASE, callback=lambda: set_command_mode("backward")),
                Keybind("mode_turn_left", Key.F10, KeyAction.RELEASE, callback=lambda: set_command_mode("turn_left")),
                Keybind("mode_turn_right", Key.F12, KeyAction.RELEASE, callback=lambda: set_command_mode("turn_right")),
                Keybind("mode_neutral", Key.F3, KeyAction.RELEASE, callback=lambda: set_command_mode("neutral")),
                Keybind("quit_eval", Key.ESCAPE, KeyAction.RELEASE, callback=stop_program),
            )

            runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
            runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
            policy = runner.get_inference_policy(device=gs.device)
            obs_dict = env.reset()

            while state["running"] and not state["restart_requested"]:
                if env.is_motion_paused:
                    actions = torch.zeros((env.num_envs, env.num_actions), dtype=gs.tc_float, device=gs.device)
                else:
                    actions = policy(obs_dict)
                obs_dict, rews, dones, infos = env.step(actions)

            env.scene.destroy()
            state["env"] = None


if __name__ == "__main__":
    main()

"""
# evaluation (turning variant)
python examples/locomotion/turning/go2_eval_turning.py -e go2-walking --ckpt 100

# keys
# F5: pause/resume motion
# F6: restart simulation
# F7: toggle camera follow
# F8: forward mode
# F9: backward mode
# F10: turn-left mode
# F12: turn-right mode
# F3: neutral (zero command)
# ESC: quit
"""

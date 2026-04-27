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

from go2_env import Go2Env


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

    state = {
        "running": True,
        "restart_requested": False,
        "env": None,
    }

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

            env.scene.viewer.register_keybinds(
                Keybind("toggle_pause_motion", Key.F5, KeyAction.RELEASE, callback=toggle_motion_pause),
                Keybind("restart_sim", Key.F6, KeyAction.RELEASE, callback=restart_simulation),
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
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking --ckpt 100
"""

"""
go1_record.py  -  Go1 simulation video recorder

Usage
-----
# 無学習（ゼロアクション）の挙動を記録
python examples/locomotion_go1/go1_record.py

# 学習済みモデル（100 イテレーション後）の挙動を記録
python examples/locomotion_go1/go1_record.py -e go1-walking -c 100

オプション
----------
-e / --exp_name   : 実験名（logs/ 以下のディレクトリ）  [default: go1-walking]
-c / --ckpt       : チェックポイント番号。省略時は無学習
-s / --num_steps  : 記録するシミュレーションステップ数  [default: 500]
-o / --output     : 出力 mp4 パス。省略時は自動生成
"""

import argparse
import copy
import os
import pickle
from datetime import datetime

import torch

import genesis as gs

from go1_env import Go1Env


def _default_cfgs():
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        "kp": 20.0,
        "kd": 0.5,
        "termination_if_roll_greater_than": 10,
        "termination_if_pitch_greater_than": 10,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {},  # 記録時はリワード不要
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_range": [0.0, 0.0],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser(description="Go1 simulation video recorder")
    parser.add_argument("-e", "--exp_name", type=str, default="go1-walking",
                        help="実験名 (logs/ 以下のディレクトリ名)")
    parser.add_argument("-c", "--ckpt", type=int, default=None,
                        help="チェックポイント番号。省略時は無学習（ゼロアクション）")
    parser.add_argument("-s", "--num_steps", type=int, default=500,
                        help="記録するシミュレーションステップ数 (default: 500)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="出力 mp4 パス (省略時は自動生成)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 出力パスの決定
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output is not None:
        video_path = args.output
    else:
        os.makedirs("videos", exist_ok=True)
        if args.ckpt is None:
            label = "untrained"
        else:
            label = f"{args.exp_name}_ckpt{args.ckpt}"
        video_path = f"videos/go1_{label}_{timestamp}.mp4"

    print(f"[record] 出力先: {video_path}")

    # ------------------------------------------------------------------
    # 設定の読み込み
    # ------------------------------------------------------------------
    log_dir = f"logs/{args.exp_name}"
    cfgs_path = os.path.join(log_dir, "cfgs.pkl")

    if args.ckpt is not None and not os.path.exists(cfgs_path):
        raise FileNotFoundError(
            f"設定ファイルが見つかりません: {cfgs_path}\n"
            f"先に go1_train.py -e {args.exp_name} で学習してください。"
        )

    if os.path.exists(cfgs_path):
        with open(cfgs_path, "rb") as f:
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
        reward_cfg["reward_scales"] = {}
        print(f"[record] 設定を {cfgs_path} から読み込みました")
    else:
        env_cfg, obs_cfg, reward_cfg, command_cfg = _default_cfgs()
        train_cfg = None
        print("[record] デフォルト設定を使用します（無学習モード）")

    # ------------------------------------------------------------------
    # Genesis 初期化
    # ------------------------------------------------------------------
    gs.init(backend=gs.cpu, logging_level="warning")

    # 無学習モードは終了閾値を実質無効化（倒れてもリセットしない）
    if args.ckpt is None:
        env_cfg["termination_if_roll_greater_than"] = 360
        env_cfg["termination_if_pitch_greater_than"] = 360

    # ------------------------------------------------------------------
    # 環境構築（1 env、カメラ付き）
    # ------------------------------------------------------------------
    env = Go1Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        record=True,
    )

    # カメラをロボット後方上方に配置（よく見えるアングル）
    env.cam.set_pose(
        pos=(3.0, 0.0, 1.5),
        lookat=(0.0, 0.0, 0.3),
    )

    # ------------------------------------------------------------------
    # ポリシーの準備
    # ------------------------------------------------------------------
    policy = None
    if args.ckpt is not None:
        try:
            from importlib import metadata
            if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
                raise ImportError
        except (metadata.PackageNotFoundError, ImportError) as e:
            raise ImportError("rsl-rl-lib>=5.0.0 をインストールしてください。") from e

        from rsl_rl.runners import OnPolicyRunner

        ckpt_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"チェックポイントが見つかりません: {ckpt_path}")

        runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir, device=gs.device)
        runner.load(ckpt_path)
        policy = runner.get_inference_policy(device=gs.device)
        print(f"[record] モデル読み込み完了: {ckpt_path}")
    else:
        # kp=kd=0 にしてPD制御を無効化 → 重力で倒れる
        env.robot.set_dofs_kp([0.0] * env.num_actions, env.motors_dof_idx)
        env.robot.set_dofs_kv([0.0] * env.num_actions, env.motors_dof_idx)
        print("[record] 無学習モード: PD制御無効（重力で倒れます）")

    # ------------------------------------------------------------------
    # シミュレーション & 録画
    # ------------------------------------------------------------------
    fps = int(1.0 / env.dt)  # 50 fps
    obs_dict = env.reset()

    # カメラ追従設定
    CAM_DIST      = 3.0   # ロボットから水平方向の距離(m)
    CAM_HEIGHT    = 1.5   # カメラ高さ(m)
    LOOKAT_HEIGHT = 0.3   # lookat の高さ(m)
    FOLLOW_THRESHOLD = 1.5   # ロボットが初期 lookat からこの距離(m)を超えたら追従開始

    init_cam_pos    = [CAM_DIST, 0.0, CAM_HEIGHT]
    init_cam_lookat = [0.0, 0.0, LOOKAT_HEIGHT]
    following = False   # 一度 True になったら以降ずっと追従

    print(f"[record] 録画開始 ({args.num_steps} ステップ = {args.num_steps * env.dt:.1f} 秒)")
    env.cam.start_recording()

    with torch.no_grad():
        for step in range(args.num_steps):
            if policy is not None:
                actions = policy(obs_dict)
            else:
                # kp=kd=0 なのでアクションはトルクに影響しない（念のためゼロ送信）
                actions = torch.zeros((env.num_envs, env.num_actions), dtype=gs.tc_float, device=gs.device)

            obs_dict, _rews, _dones, _infos = env.step(actions)

            # ── カメラ追従 ──────────────────────────────────────────
            robot_pos = env.robot.get_pos(envs_idx=[0])[0].cpu()  # (3,)
            rx, ry = robot_pos[0].item(), robot_pos[1].item()

            if not following:
                dist = (rx ** 2 + ry ** 2) ** 0.5
                if dist > FOLLOW_THRESHOLD:
                    following = True

            if following:
                # lookat = ロボットの真上
                lookat = [rx, ry, LOOKAT_HEIGHT]
                # カメラは現在の lookat からロボットの方向と逆側（背後）に配置
                # ロボット進行方向を前回 pos との差で取れないため、
                # ここでは常にロボットの正面 (-X 方向) から見る固定アングルで追従
                # ※ ロボットの向きに合わせたい場合は base_quat から計算可
                cam_pos = [rx + CAM_DIST, ry, CAM_HEIGHT]
                env.cam.set_pose(pos=cam_pos, lookat=lookat)
            # ────────────────────────────────────────────────────────

            env.cam.render()

            if (step + 1) % 100 == 0:
                print(f"[record]   step {step + 1}/{args.num_steps}")

    env.cam.stop_recording(save_to_filename=video_path, fps=fps)
    env.scene.destroy()

    print(f"[record] 録画完了 -> {video_path}")


if __name__ == "__main__":
    main()

"""
# ── 使い方 ──────────────────────────────────────────────────────────────

# 1. 無学習の挙動を記録
python examples/locomotion_go1/go1_record.py

# 2. 学習（4096並列, 24ステップ, 100回）してから結果を記録
python examples/locomotion_go1/go1_train.py -e go1-test -B 4096 --max_iterations 101
python examples/locomotion_go1/go1_record.py -e go1-test -c 100

# 3. ステップ数や出力先を指定
python examples/locomotion_go1/go1_record.py --num_steps 300 --output videos/my_test.mp4
"""

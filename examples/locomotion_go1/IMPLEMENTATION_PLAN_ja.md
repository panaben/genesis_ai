# Go1 Locomotion 実装計画

## 目的

既存の Go2 locomotion サンプルをベースに、Go1 版を `examples/locomotion_go1` 配下に作成する。

作成予定ファイル:

- `go1_env.py`
- `go1_train.py`
- `go1_eval.py`

最初の目標は歩行性能の高さではなく、Go1 用の学習・評価ループが安定して起動し、`build`、`step`、短時間学習、checkpoint 保存、checkpoint 評価まで通ること。

## 現在の実現可能性確認結果

`genesis/assets/urdf/go1_description/urdf/go1.urdf` は Genesis でロードできる。

`check_go1_urdf.py` の確認結果:

- `scene.build(n_envs=1)` が成功。
- `scene.step()` が 10 step 成功。
- `n_dofs = 18`。内訳は floating base 6 DOF + motor 12 DOF。
- Go1 に必要な 12 個の motor joint が存在する:
  - `FR_hip_joint`, `FR_thigh_joint`, `FR_calf_joint`
  - `FL_hip_joint`, `FL_thigh_joint`, `FL_calf_joint`
  - `RR_hip_joint`, `RR_thigh_joint`, `RR_calf_joint`
  - `RL_hip_joint`, `RL_thigh_joint`, `RL_calf_joint`

既知の warning:

```text
Neutral robot position (qpos0) exceeds joint limits.
```

これは URDF の neutral joint value が 0 である一方、calf joint の有効範囲が負の角度であるため発生していると考えられる。locomotion env では reset 時に `default_joint_angles` を設定し、calf joint には有効な負の値を入れる想定なので、runtime reset や最初の simulation step で不安定にならない限り許容する。

## Go1 と Go2 の主な違い

URDF からの静的比較:

| 項目 | Go2 | Go1 | 影響 |
| --- | ---: | ---: | --- |
| URDF inertial から見た総質量 | 約 15.019 kg | 約 13.101 kg | PD ゲインや reward scale の調整が必要になる可能性がある。 |
| fixed-link merge 前の link 数 | 29 | 46 | Go1 は sensor / rotor 系の fixed link が多い。ただし Genesis 側で merge される。 |
| fixed-link merge 前の joint 数 | 28 | 45 | 実行確認では Go1 は merge 後 `n_joints = 13`。 |
| hip limit 例 | `[-1.0472, 1.0472]` | `[-0.863, 0.863]` | Go1 の hip 可動域は Go2 より狭い。 |
| thigh limit 例 | `[-1.5708, 3.4907]` | `[-0.686, 4.501]` | Go1 の thigh 可動域は Go2 と少しずれている。 |
| calf limit 例 | `[-2.7227, -0.83776]` | `[-2.818, -0.888]` | 近いが、default pose は必ず limit 内に置く必要がある。 |

Go2 の checkpoint を Go1 にそのまま使えるとは考えない。action / observation の次元は一致するが、質量、関節制限、形状、動力学、reward target が異なる。

## Phase 0 - URDF 確認スクリプトを残す

対象ファイル:

- `check_go1_urdf.py`

目的:

- Go1 asset 自体の smoke test として残す。
- URDF や mesh を変更した場合は、env / training コードより先にこのスクリプトを実行する。

実行コマンド:

```bash
python examples/locomotion_go1/check_go1_urdf.py
```

合格条件:

- 最後に `GO1 URDF load/build/step OK` が出る。
- 12 個の required motor joints がすべて表示される。

## Phase 1 - `go1_env.py` を作成する

`examples/locomotion/go2_env.py` をベースにする。ただし Go1 固有の部分は明示する。

必要な変更:

- class 名を `Go2Env` から `Go1Env` に変更する。
- robot URDF を `urdf/go1_description/urdf/go1.urdf` に変更する。
- 12 action interface と `joint_names` の契約は維持する。
- `actions_dof_idx = torch.argsort(self.motors_dof_idx)` は維持する。Go1 の runtime DOF order は hips、thighs、calves のように並ぶため、action order と DOF order の対応補正が必要。
- `control_dofs_position(..., slice(6, 18))` は `n_dofs == 18` を確認したうえで維持する。
- `scene.build()` 後に小さな validation を追加する:
  - 設定された motor joints がすべて存在すること。
  - `robot.n_dofs == 18` であること。
  - 可能なら `default_joint_angles` が joint limit 内にあること。

初期 config 方針:

- まず Go2 の設定を baseline として使う:
  - `base_init_pos = [0.0, 0.0, 0.42]`
  - hip default `0.0`
  - front thigh `0.8`
  - rear thigh `1.0`
  - calf `-1.5`
  - `kp = 20.0`
  - `kd = 0.5`
  - `action_scale = 0.25`
- sanity test で不安定性が見えた場合だけ調整する。

Phase 1 後の確認:

1. CPU で `Go1Env(num_envs=1)` を作る。
2. `reset()` を呼ぶ。
3. zero action で 100-300 step 進める。
4. 以下を確認する:
   - rigid solver error envs が出ない。
   - base が即座に暴れない。
   - reset buffer が常に true にならない。
   - base height が妥当に見える。

PD ゲインや base height が本当に課題になるかは、この段階で初めて判断する。

## Phase 2 - `go1_train.py` を作成する

`examples/locomotion/go2_train.py` をベースにする。

必要な変更:

- `go1_env` から `Go1Env` を import する。
- default experiment name を `go1-walking` にする。
- PPO config は最初は変更しない。
- `num_actions = 12` を維持する。
- command の形も維持する:
  - `num_commands = 3`
  - x velocity, y velocity, yaw velocity

最初の smoke training:

```bash
python examples/locomotion_go1/go1_train.py -e go1-walking-smoke -B 64 --max_iterations 2
```

smoke test 合格条件:

- training が開始する。
- rollout 中に例外が出ない。
- `logs/go1-walking-smoke/cfgs.pkl` が作成される。
- 少なくとも 1 iteration 完了する。

その後、規模を上げる:

```bash
python examples/locomotion_go1/go1_train.py -e go1-walking -B 4096 --max_iterations 101
```

## Phase 3 - Reward と PD の調整確認

一度に全部を調整しない。段階ごとに確認する。

### Checkpoint A - Default Pose と PD Hold

タイミング:

- `go1_env.py` 作成直後。

実行内容:

- `num_envs=1`
- zero actions
- learning なし

見るもの:

- Go1 が default pose 付近で立てるか。
- 強い jitter が出るか。
- command や reward 以前に倒れるか。

必要なら調整:

- jitter が強い場合: `kp` を下げる、または `kd` を慎重に調整する。
- 関節が弱く崩れる場合: `kp` を少し上げる。
- 制御が強すぎる場合: reward より先に `action_scale` を下げる。

初期 PD baseline:

- `kp = 20.0`
- `kd = 0.5`

候補レンジ:

- `kp`: 15.0 から 30.0
- `kd`: 0.4 から 1.0
- `action_scale`: 0.15 から 0.25

### Checkpoint B - Base Height Target

タイミング:

- zero-action hold が安定した後。

現在の Go2 reward target:

- `base_height_target = 0.3`

Go1 で確認すること:

- 安定姿勢での base height が `0.28`、`0.30`、または別の値のどれに近いか。

方針:

- zero-action hold 中の平均 base height を log または print する。
- 観測された安定姿勢の高さに近い値へ `base_height_target` を設定する。
- penalty weight は最初は変更しない: `base_height = -50.0`

### Checkpoint C - 短時間 PPO Training

タイミング:

- PD hold と base height が妥当に見えた後。

実行:

```bash
python examples/locomotion_go1/go1_train.py -e go1-walking-test -B 512 --max_iterations 20
```

見るもの:

- reward trend
- episode length
- reset frequency
- `rew_tracking_lin_vel`
- `rew_base_height`
- `rew_similar_to_default`
- solver error reset

必要なら調整:

- しゃがむ、跳ねるなど reward hack が出る場合: `base_height_target` または `base_height` penalty を調整する。
- default pose 付近で固まる場合: `similar_to_default` penalty の絶対値を下げる。
- action が noisy な場合: `action_rate` penalty の絶対値を上げる、または `action_scale` を下げる。
- velocity tracking ができない場合: reward より先に PD を確認し、その後 tracking reward を見る。

### Checkpoint D - Full Training

タイミング:

- 短時間 PPO で破綻しない動きが出た後。

実行:

```bash
python examples/locomotion_go1/go1_train.py -e go1-walking -B 4096 --max_iterations 101
```

policy quality の判断はこの段階以降に行う。

## Phase 4 - `go1_eval.py` を作成する

`examples/locomotion/go2_eval.py` をベースにする。

必要な変更:

- `Go1Env` を import する。
- default experiment name を `go1-walking` にする。
- checkpoint は `logs/<exp_name>/model_<ckpt>.pt` から読み込む。
- pause / restart / camera follow の viewer control は維持する。

評価コマンド:

```bash
python examples/locomotion_go1/go1_eval.py -e go1-walking --ckpt 100
```

合格条件:

- viewer が起動する。
- checkpoint が読み込まれる。
- shape mismatch なしで policy step が進む。
- Go1 が即座に solver failure にならず動く。

## 実装順序

1. `check_go1_urdf.py` を asset smoke test として維持する。
2. Go1 URDF と validation を入れた `go1_env.py` を追加する。
3. 必要なら env smoke test 用コマンドまたは script を追加する。
4. `go1_train.py` を追加する。
5. `-B 64 --max_iterations 2` を実行する。
6. PD / default pose / base height を確認する。
7. `-B 512 --max_iterations 20` を実行する。
8. 短時間学習の観測結果を見てから reward / PD を調整する。
9. `go1_eval.py` を追加する。
10. 学習済み checkpoint で evaluation を実行する。

## 初期 Go / No-Go 判定

Go:

- `check_go1_urdf.py` が通る。
- `Go1Env` が `n_dofs == 18` で build できる。
- zero-action stepping が PPO smoke training を始められる程度に安定している。
- PPO smoke training が少なくとも 2 iteration 完了する。

No-go / 要調査:

- required joint names が見つからない。
- runtime DOF count が 18 ではない。
- default joint pose が joint limit を外れる。
- zero actions で即座に solver error envs が出る。
- reward tuning を始める前に robot が即座に倒れる。


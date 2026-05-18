# Go1 Sim2Real2Sim ワークフロー

Genesis でポリシーを学習し、実機で動かし、実機データを使ってシミュレーターを補正して再学習するまでの手順書。

```
[STEP 1] シミュレーター学習  →  model_N.pt
    ↓
[STEP 2] シミュレーター評価  →  動作確認
    ↓
[STEP 3] 実機デプロイ        →  deploy_logs/run_*.csv
    ↓
[STEP 4] ログ解析            →  .train_env (パラメータ調整値)
    ↓
[STEP 5] 再学習              →  model_M.pt (Sim2Real補正済み)
    ↓
[STEP 3] へ戻る（繰り返し）
```

---

## STEP 1: シミュレーター学習

```bash
# リポジトリルートから実行
python examples/locomotion_go1/go1_train.py \
    -e go1-walking \
    -B 4096 \
    --max_iterations 101
```

| 引数 | 説明 |
|---|---|
| `-e` | 実験名。`logs/<exp_name>/` に保存される |
| `-B` | 並列環境数。GPU メモリに応じて調整 |
| `--max_iterations` | 学習イテレーション数 |

**出力:**
```
logs/go1-walking/
    cfgs.pkl        ← 設定のスナップショット
    model_0.pt
    model_100.pt    ← チェックポイント（save_interval=100 ごと）
```

---

## STEP 2: シミュレーター評価

実機に繋ぐ前に必ず実施する。

```bash
python examples/locomotion_go1/go1_eval.py \
    -e go1-walking \
    --ckpt 100
```

| キー | 動作 |
|---|---|
| `F5` | モーション一時停止 / 再開 |
| `F6` | シミュレーション再起動 |
| `F7` | カメラフォロー切り替え |
| `ESC` | 終了 |

歩行が安定していることを確認してから STEP 3 へ。

---

## STEP 3: 実機デプロイ（Real）

> **詳細手順は [deploy/README.md](deploy/README.md) を参照。**

### 前提

- **Linux のみ**（`robot_interface.so` は Linux 専用）
- Ethernet で Go1 と同一サブネット（`192.168.123.x`）に接続済み
- Go1 を **DAMPING モード**（リモコン: L2 + A）にしてからスクリプトを起動

### ⚠️ 安全上の注意

> **怠ると機体の転倒・破損につながります。**

#### スタートアップ時の体勢

スクリプト起動直後、現在の関節角度から立ち姿勢まで **2秒かけて補間移動**します。
この間に脚が予期しない方向へ動くため、以下のいずれかの状態でスタートしてください。

| 方法 | 詳細 |
|---|---|
| **吊るす（推奨）** | フレームにハーネス等を掛けて脚が床に届かない状態にする |
| **平地に置く** | 平坦な床に置き、人が横で支えられる体制を取る |

#### 立ち上がり確認後にポリシーを開始する

スクリプトは Enter キーを2回待ちます。**脚が立ち姿勢に整ったことを目視確認してから** 2回目の Enter を押してください。

```
[deploy] *** Verify the robot is suspended or on flat ground. ***
         Press Enter to begin STANDUP sequence ...    ← 1回目: 吊るした状態で押す

[deploy] Standing up (2.0 s, 100 steps) ...
[deploy] Standup complete.
[deploy] >>> Press Enter to start WALKING policy ...  ← 2回目: 姿勢確認後に押す
```

#### 緊急停止

| トリガー | 動作 |
|---|---|
| `Ctrl + C` | 即時停止 → ダンピングモードへ移行 |
| roll > ±30° または pitch > ±30° | 自動緊急停止 → ダンピングモードへ移行 |

停止後は全モータが `Kp=0, Kd=2` のダンピング状態になります（受動的に脱力）。

### 実行コマンド

```bash
# Linux 環境・Ethernet 接続済みの状態で実行（sudo 必要）
sudo python examples/locomotion_go1/deploy/go1_deploy.py \
    --model_dir logs/go1-walking \
    --ckpt 100 \
    --vx 0.3 --vy 0.0 --wz 0.0 \
    --log
```

| オプション | デフォルト | 説明 |
|---|---|---|
| `--model_dir` | `logs/go1-walking` | `cfgs.pkl` と `model_N.pt` があるディレクトリ |
| `--ckpt` | `100` | チェックポイント番号 |
| `--vx` | `0.3` | 前進速度指令 [m/s] |
| `--vy` | `0.0` | 横方向速度指令 [m/s] |
| `--wz` | `0.0` | ヨー角速度指令 [rad/s] |
| `--log` | （無効） | CSV ログを有効化 |
| `--log_dir` | `deploy_logs` | ログ出力先ディレクトリ |

`--log` を付けると `deploy_logs/run_<timestamp>.csv` が生成される。これが STEP 4 の入力。

**CSV の内容（56列）:**

```
timestamp, dof_pos[12], dof_vel[12], tau_est[12],
imu_gyro[3], imu_quat[4], cmd[3], action[12]
```

---

## STEP 4: ログ解析 → `.train_env` 生成

CSV から実機とシミュレーターのギャップを定量化し、`.train_env` に書き出す。

```bash
# analyze_logs.py は deploy/retrain/ に配置予定（現在開発中）
# 手動で .train_env を編集する場合は .train_env.example をコピーして使う
cp examples/locomotion_go1/.train_env.example examples/locomotion_go1/.train_env
```

`.train_env` の編集ポイント:

```toml
[training]
resume_ckpt    = "logs/go1-walking/model_100.pt"  # 継続元のチェックポイント
num_envs       = 4096
max_iterations = 200  # 追加学習イテレーション数

[env_cfg]
# 実機の応答からフィッティングした PD ゲイン
kp = 18.0   # 実機が柔らかければ下げる
kd = 0.55   # 振動が多ければ上げる

[reward_scales]
# 実機で不安定だった動作に対応するペナルティ強化
action_rate = -0.01   # 急激なアクション変化を抑制

[command_cfg]
# 実機で試した速度レンジに絞ることで転移しやすくする
lin_vel_x_range = [0.2, 0.4]
```

### `.train_env` の全セクション

| セクション | 上書き先 | 主な用途 |
|---|---|---|
| `[training]` | runner 制御 | resume_ckpt, イテレーション数 |
| `[env_cfg]` | `env_cfg` | kp, kd, 終了条件 |
| `[obs_scales]` | `obs_cfg["obs_scales"]` | 観測正規化スケール |
| `[reward_scales]` | `reward_cfg["reward_scales"]` | 報酬重み |
| `[command_cfg]` | `command_cfg` | 速度指令レンジ |

指定しなかったキーはすべて `get_cfgs()` のデフォルト値が使われる。

---

## STEP 5: 再学習（Sim with Real-adjusted params）

```bash
python examples/locomotion_go1/go1_train.py \
    -e go1-walking-v2 \
    --env_file examples/locomotion_go1/.train_env
```

`resume_ckpt` が `.train_env` に指定されている場合、`logs/go1-walking/` は **削除されず**、重みを引き継いで学習が継続される。新しい実験名 (`-e go1-walking-v2`) を指定することで別ディレクトリに保存される。

**出力:**
```
logs/go1-walking-v2/
    cfgs.pkl
    model_100.pt
    model_200.pt
    ...
```

STEP 2 でシミュレーター評価 → STEP 3 で実機評価 → を繰り返す。

---

## パラメータ調整の指針

### 実機でよく見られる症状と対応

| 症状 | 疑わしい原因 | `.train_env` での対処 |
|---|---|---|
| 脚がブルブル振動する | Kd が低い / シミュレーターより実機の減衰が小さい | `kd` を上げる（例: 0.5 → 0.7） |
| 指令速度に追従できない | Kp が低い / アクションスケールのずれ | `kp` を上げる、`reward_scales.tracking_lin_vel` を増やす |
| 急激な姿勢変化で転倒 | アクション変化が大きい | `reward_scales.action_rate` のペナルティを強化 |
| 低速でしか安定しない | 学習速度レンジが広すぎた | `command_cfg.lin_vel_x_range` を実機で安定した範囲に絞る |
| 体が沈む | 実機の重量/慣性がシミュレーターと違う | `reward_scales.base_height` のペナルティを強化 |

---

## ファイル構成

```
examples/locomotion_go1/
    go1_train.py          学習スクリプト（--env_file 対応済み）
    go1_eval.py           シミュレーター評価スクリプト
    go1_env.py            Genesis 学習環境（PPO 用）
    .train_env.example    .train_env のサンプル
    deploy/
        go1_deploy.py     実機デプロイスクリプト（50 Hz Low-level 制御）
        go1_logger.py     非同期 CSV ロガー
        README.md         デプロイ詳細手順

logs/
    go1-walking/
        cfgs.pkl
        model_100.pt      ← STEP 3 で使う
    go1-walking-v2/
        model_200.pt      ← STEP 5 の出力

deploy_logs/
    run_<timestamp>.csv   ← STEP 3 の出力 / STEP 4 の入力
```

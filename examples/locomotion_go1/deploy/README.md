# Go1 実機デプロイ手順

学習済みポリシー（`model_N.pt` + `cfgs.pkl`）を使って Unitree Go1 実機を歩かせるための手順書です。

---

## ⚠️ 前提条件

### 動作環境

| 項目 | 要件 |
|---|---|
| OS | **Linux のみ**（Ubuntu 18.04 / 20.04 推奨） |
| Python | ビルドした `.so` に合わせること（例: Python 3.11） |
| アーキテクチャ | x86_64 または ARM64 |
| ネットワーク | Ethernet で Go1 と同一サブネット（`192.168.123.x`） |

> Windows では `robot_interface.so`（Linux 共有ライブラリ）が動作しないため**使用不可**です。

---

## 1. 事前準備

### 1-1. SDKのビルド（初回のみ）

```bash
cd <repo_root>/third_party/unitree_legged_sdk
mkdir -p build && cd build
cmake -DPYTHON_BUILD=TRUE ..
make
```

ビルド後、`lib/python/amd64/robot_interface.cpython-3XX-x86_64-linux-gnu.so` が生成されていることを確認してください。

### 1-2. Python パッケージのインストール

```bash
pip install "rsl-rl-lib>=5.0.0" tensordict torch
```

### 1-3. 必要ファイルの確認

デプロイに必要なファイルが揃っていることを確認します。

```
logs/go1-walking/
    cfgs.pkl        ← 学習時の設定（必須）
    model_100.pt    ← 学習済みモデル（必須）
```

---

## 2. シミュレーションでの事前確認（推奨）

実機に繋ぐ前に、シミュレーターで動作確認してください。

```bash
# リポジトリルートから実行
python examples/locomotion_go1/go1_eval.py -e go1-walking --ckpt 100
```

| キー | 動作 |
|---|---|
| `F5` | モーション一時停止 / 再開 |
| `F6` | シミュレーション再起動 |
| `F7` | カメラフォロー切り替え |
| `ESC` | 終了 |

---

## 3. 実機接続の準備

### 3-1. ネットワーク設定

PC の Ethernet アダプタを以下に設定します。

```
IPアドレス : 192.168.123.xxx  (xxx は 10 以外の任意)
サブネット  : 255.255.255.0
```

ロボット側の IP は `192.168.123.10`（固定）です。

### 3-2. Go1 を DAMPING モードに設定

付属のリモコンで Go1 を **DAMPING モード（脱力状態）** にしてから、スクリプトを実行してください。

> DAMPING モード: L2 + A ボタン（機種・ファームウェアによって異なる場合があります）

---

## 4. 安全上の注意

> **以下を必ず守ってください。怠ると機体の転倒・破損につながります。**

### スタートアップ時の体勢

スクリプト起動直後、現在の関節角度から立ち姿勢 `DEFAULT_DOF_POS` まで **2秒かけて補間移動**します。
この間に脚が予期しない方向へ動くため、以下のいずれかの状態でスタートしてください。

| 方法 | 詳細 |
|---|---|
| **吊るす（推奨）** | フレームにハーネス等を掛けて脚が床に届かない状態にする |
| **平地に置く** | 平坦な床に置き、人が横で支えられる体制を取る |

### 立ち上がり確認後にポリシーを開始する

スクリプトは Enter キーを2回待ちます。**脚が立ち姿勢に整ったことを目視確認してから** 2回目の Enter を押してください。

```
[deploy] *** Verify the robot is suspended or on flat ground. ***
         Press Enter to begin STANDUP sequence ...    ← 1回目: 吊るした状態で押す

[deploy] Standing up (2.0 s, 100 steps) ...
[deploy] Standup complete.
[deploy] >>> Press Enter to start WALKING policy ...  ← 2回目: 姿勢確認後に押す
```

### 緊急停止

| トリガー | 動作 |
|---|---|
| `Ctrl + C` | 即時停止 → ダンピングモードへ移行 |
| roll > ±30° または pitch > ±30° | 自動緊急停止 → ダンピングモードへ移行 |

停止後は全モータが `Kp=0, Kd=2` のダンピング状態になります（受動的に脱力）。

---

## 5. 実行コマンド

```bash
# リポジトリルートから実行（sudo 必要）
sudo python examples/locomotion_go1/deploy/go1_deploy.py \
    --model_dir logs/go1-walking \
    --ckpt 100 \
    --vx 0.3 --vy 0.0 --wz 0.0
```

### オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--model_dir` | `logs/go1-walking` | `cfgs.pkl` と `model_N.pt` があるディレクトリ |
| `--ckpt` | `100` | チェックポイント番号（`model_100.pt` を使う場合は `100`） |
| `--vx` | `0.3` | 前進速度指令 [m/s]（正 = 前進） |
| `--vy` | `0.0` | 横方向速度指令 [m/s]（正 = 左） |
| `--wz` | `0.0` | ヨー角速度指令 [rad/s]（正 = 左旋回） |
| `--log` | （無効） | CSV ログを有効化 |
| `--log_dir` | `deploy_logs` | ログ出力先ディレクトリ |

### 実行例

```bash
# 前進 0.5 m/s でログあり
sudo python examples/locomotion_go1/deploy/go1_deploy.py \
    --model_dir logs/go1-walking --ckpt 100 --vx 0.5 --log

# 左旋回
sudo python examples/locomotion_go1/deploy/go1_deploy.py \
    --model_dir logs/go1-walking --ckpt 100 --vx 0.2 --wz 0.5
```

---

## 6. ログの活用

`--log` を指定すると `deploy_logs/run_<timestamp>.csv` に記録されます。

**記録される列（合計 56 列）:**

```
timestamp,
dof_pos_0..11,    # 12関節の角度 [rad]
dof_vel_0..11,    # 12関節の角速度 [rad/s]
tau_est_0..11,    # 推定トルク [Nm]
imu_gyro_0..2,    # IMU ジャイロ (body frame) [rad/s]
imu_quat_0..3,    # IMU クォータニオン [w, x, y, z]
cmd_0..2,         # 速度指令 [vx, vy, wz]
action_0..11      # ポリシーが出力したアクション
```

---

## 7. 制御パラメータ

| パラメータ | 値 | 用途 |
|---|---|---|
| 制御周期 | 50 Hz (20 ms) | メインループ |
| スタンドアップ Kp / Kd | 5.0 / 1.0 | 低ゲインで安全に立ち上げ |
| 歩行 Kp / Kd | 20.0 / 0.5 | 学習時と同じ値 |
| ダンピング Kp / Kd | 0.0 / 2.0 | 停止時の脱力 |
| アクションスケール | 0.25 | `target_q = action * 0.25 + default_q` |

---

## 8. モータインデックス対応表

```
インデックス  関節名
  0         FR_hip    (右前・ヒップ)
  1         FR_thigh  (右前・太もも)
  2         FR_calf   (右前・ふくらはぎ)
  3         FL_hip    (左前・ヒップ)
  4         FL_thigh  (左前・太もも)
  5         FL_calf   (左前・ふくらはぎ)
  6         RR_hip    (右後・ヒップ)
  7         RR_thigh  (右後・太もも)
  8         RR_calf   (右後・ふくらはぎ)
  9         RL_hip    (左後・ヒップ)
 10         RL_thigh  (左後・太もも)
 11         RL_calf   (左後・ふくらはぎ)
```

---

## 関連ファイル

| ファイル | 説明 |
|---|---|
| [go1_deploy.py](go1_deploy.py) | メインのデプロイスクリプト |
| [go1_logger.py](go1_logger.py) | 非同期 CSV ロガー |
| [../go1_train.py](../go1_train.py) | 学習スクリプト |
| [../go1_eval.py](../go1_eval.py) | シミュレーション評価スクリプト |
| [../go1_env.py](../go1_env.py) | 学習環境（観測・報酬定義） |

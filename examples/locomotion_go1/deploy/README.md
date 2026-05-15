# Go1 実機デプロイ手順

学習済みポリシー（`model_N.pt` + `cfgs.pkl`）を使って Unitree Go1 実機を歩かせるための手順書です。

---

## ⚠️ 前提条件

### 動作環境

| 項目 | 要件 |
|---|---|
| OS | **Linux のみ**（Ubuntu 18.04 / 20.04 推奨） |
| Python | **3.8 固定**（`robot_interface.cpython-38-*.so` のため） |
| アーキテクチャ | x86_64 または ARM64 |
| ネットワーク | Ethernet で Go1 と同一サブネット（`192.168.123.x`） |

> Windows では `robot_interface.so`（Linux共有ライブラリ）が動作しないため**使用不可**です。

---

## 1. 事前準備

### 1-1. SDKのビルド（初回のみ）

```bash
cd <repo_root>/third_party/unitree_legged_sdk
mkdir -p build && cd build
cmake -DPYTHON_BUILD=TRUE ..
make
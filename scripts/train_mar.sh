#!/bin/bash

# GPUリストと対応する設定ファイルのリストを定義
CONFIG_FILES=(
  "./configs/exp_mar_adaln_enet/ad_mar_bottle.yml"
  "./configs/exp_mar_adaln_enet/ad_mar_cable.yml"
  "./configs/exp_mar_adaln_enet/ad_mar_capsule.yml"
  "./configs/exp_mar_adaln_enet/ad_mar_carpet.yml"
  "./configs/exp_mar_adaln_enet/ad_mar_grid.yml"
  "./configs/exp_mar_adaln_enet/ad_mar_hazelnut.yml"
  "./configs/exp_mar_adaln_enet/ad_mar_leather.yml"
  "./configs/exp_mar_adaln_enet/ad_mar_metal_nut.yml"
  "./configs/exp_mar_adaln_enet/ad_mar_pill.yml"
  "./configs/exp_mar_adaln_enet/ad_mar_screw.yml"
  "./configs/exp_mar_adaln_enet/ad_mar_tile.yml"
  "./configs/exp_mar_adaln_enet/ad_mar_toothbrush.yml"
  "./configs/exp_mar_adaln_enet/ad_mar_transistor.yml"
  "./configs/exp_mar_adaln_enet/ad_mar_wood.yml"
  "./configs/exp_mar_adaln_enet/ad_mar_zipper.yml"
)

# 最大GPU数を指定
MAX_GPUS=6

# メモリ使用量の上限 (1GB = 1024MB)
MEMORY_LIMIT_MB=1024

# GPUの空きを待つ関数
wait_for_free_gpu() {
  while true; do
    for ((i=0; i<$MAX_GPUS; i++)); do
      # i番目のGPUの使用状況（メモリ使用量）を取得
      usage=$(nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader | sed -n "$((i+1))p")

      # usageが数値として1024MB以下なら利用可能と判断
      if [ "$usage" -le "$MEMORY_LIMIT_MB" ]; then
        echo "$i"
        return
      fi
    done
    # いずれのGPUも空いていなければ1秒待って再チェック
    sleep 1
  done
}

for CONFIG_PATH in "${CONFIG_FILES[@]}"; do
  GPU_ID=$(wait_for_free_gpu)
  echo "Using GPU $GPU_ID for config: $CONFIG_PATH"

  # バックグラウンドでジョブを開始
  CUDA_VISIBLE_DEVICES=$GPU_ID python3 ./src/train_mar.py --config_path "$CONFIG_PATH" &

  # ★ ここで1分待機してから次のジョブを投げる
  sleep 60
done

# すべてのジョブが完了するのを待つ
wait

echo "All jobs finished."

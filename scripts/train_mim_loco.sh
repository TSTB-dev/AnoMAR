#!/bin/bash

# breakfast_box      pushpins    screw_bag
# juice_bottle splicing_connectors

# GPUリストと対応する設定ファイルのリストを定義
CONFIG_FILES=(
  "./configs/exp_mim_enet_loco/loco_mim_breakfast_box.yml"
  "./configs/exp_mim_enet_loco/loco_mim_juice_bottle.yml"
  "./configs/exp_mim_enet_loco/loco_mim_pushpins.yml"
  "./configs/exp_mim_enet_loco/loco_mim_screw_bag.yml"
  "./configs/exp_mim_enet_loco/loco_mim_splicing_connectors.yml"
)

# 最大GPU数を指定
MAX_GPUS=5

# メモリ使用量の上限 (1GB = 1024MB)
MEMORY_LIMIT_MB=1024

# GPUの空きを待つ関数
wait_for_free_gpu() {
  while true; do
    for ((i=3; i<$MAX_GPUS+3; i++)); do
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
  CUDA_VISIBLE_DEVICES=$GPU_ID python3 ./src/train_mim.py --config_path "$CONFIG_PATH" &

  # ★ ここで1分待機してから次のジョブを投げる
  sleep 60
done

# すべてのジョブが完了するのを待つ
wait

echo "All jobs finished."

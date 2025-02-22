#!/bin/bash

# GPUリストを定義
GPU_LIST=(0 1 2 3 4 5 6 7)

# GPUリストと対応する設定ファイルのリストを定義
CONFIG_FILES=(
  "configs/exp_unet_feature_ad/bottle.yml"
  "configs/exp_unet_feature_ad/capsule.yml"
  "configs/exp_unet_feature_ad/grid.yml"
  "configs/exp_unet_feature_ad/leather.yml"
  "configs/exp_unet_feature_ad/metal_nut.yml"
  "configs/exp_unet_feature_ad/pill.yml"
  "configs/exp_unet_feature_ad/screw.yml"
  "configs/exp_unet_feature_ad/toothbrush.yml"
  "configs/exp_unet_feature_ad/wood.yml"
  "configs/exp_unet_feature_ad/cable.yml"
  "configs/exp_unet_feature_ad/carpet.yml"
  "configs/exp_unet_feature_ad/hazelnut.yml"
  "configs/exp_unet_feature_ad/tile.yml"
  "configs/exp_unet_feature_ad/transistor.yml"
  "configs/exp_unet_feature_ad/zipper.yml"
)

# メモリ使用量の上限 (1GB = 1024MB)
# 140 GB => 143360 MB
MEMORY_LIMIT_MB=15000

# GPUの空きを待つ関数
wait_for_free_gpu() {
  while true; do
    for GPU_ID in "${GPU_LIST[@]}"; do
      usage=$(nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader | sed -n "$((GPU_ID+1))p")
      if [ "$usage" -le "$MEMORY_LIMIT_MB" ]; then
        echo "$GPU_ID"
        return
      fi
    done
    sleep 1
  done
}

for CONFIG_PATH in "${CONFIG_FILES[@]}"; do
  GPU_ID=$(wait_for_free_gpu)
  echo "Using GPU $GPU_ID for config: $CONFIG_PATH"
  
  CUDA_VISIBLE_DEVICES=$GPU_ID python3 ./src/train_dit_feature.py --config_path "$CONFIG_PATH" &
  
  sleep 60
done

wait
echo "All jobs finished."

#!/bin/bash

# GPUリストを定義
GPU_LIST=(0 1 2 3 4 5 6 7)

# GPUリストと対応する設定ファイルのリストを定義
CONFIG_FILES=(
  "configs/exp_unet_vae_ad_ddad/bottle.yml"
  "configs/exp_unet_vae_ad_ddad/capsule.yml"
  "configs/exp_unet_vae_ad_ddad/grid.yml"
  "configs/exp_unet_vae_ad_ddad/leather.yml"
  "configs/exp_unet_vae_ad_ddad/metal_nut.yml"
  "configs/exp_unet_vae_ad_ddad/pill.yml"
  "configs/exp_unet_vae_ad_ddad/screw.yml"
  "configs/exp_unet_vae_ad_ddad/toothbrush.yml"
  "configs/exp_unet_vae_ad_ddad/wood.yml"
  "configs/exp_unet_vae_ad_ddad/cable.yml"
  "configs/exp_unet_vae_ad_ddad/carpet.yml"
  "configs/exp_unet_vae_ad_ddad/hazelnut.yml"
  "configs/exp_unet_vae_ad_ddad/tile.yml"
  "configs/exp_unet_vae_ad_ddad/transistor.yml"
  "configs/exp_unet_vae_ad_ddad/zipper.yml"
)

# メモリ使用量の上限 (1GB = 1024MB)
MEMORY_LIMIT_MB=3000

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
  
  CUDA_VISIBLE_DEVICES=$GPU_ID python3 ./src/train_dit.py --config_path "$CONFIG_PATH" &
  
  sleep 60
done

wait
echo "All jobs finished."

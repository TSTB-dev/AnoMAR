#!/bin/bash

# bottle  capsule  grid      leather      metal_nut            pill        screw  toothbrush  wood
# cable   carpet   hazelnut   tile   transistor  zipper

# GPUリストと対応する設定ファイルのリストを定義
CONFIG_FILES=(
  "./configs/exp_emar_ca_vae_ad_enc_id/ad_dit_bottle.yml"
  "./configs/exp_emar_ca_vae_ad_enc_id/ad_dit_capsule.yml"
  "./configs/exp_emar_ca_vae_ad_enc_id/ad_dit_grid.yml"
  "./configs/exp_emar_ca_vae_ad_enc_id/ad_dit_leather.yml"
  "./configs/exp_emar_ca_vae_ad_enc_id/ad_dit_metal_nut.yml"
  "./configs/exp_emar_ca_vae_ad_enc_id/ad_dit_pill.yml"
  "./configs/exp_emar_ca_vae_ad_enc_id/ad_dit_screw.yml"
  "./configs/exp_emar_ca_vae_ad_enc_id/ad_dit_toothbrush.yml"
  "./configs/exp_emar_ca_vae_ad_enc_id/ad_dit_wood.yml"
  "./configs/exp_emar_ca_vae_ad_enc_id/ad_dit_cable.yml"
  "./configs/exp_emar_ca_vae_ad_enc_id/ad_dit_carpet.yml"
  "./configs/exp_emar_ca_vae_ad_enc_id/ad_dit_hazelnut.yml"
  "./configs/exp_emar_ca_vae_ad_enc_id/ad_dit_tile.yml"
  "./configs/exp_emar_ca_vae_ad_enc_id/ad_dit_transistor.yml"
  "./configs/exp_emar_ca_vae_ad_enc_id/ad_dit_zipper.yml"
)

# 最大GPU数を指定
MAX_GPUS=8

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
  CUDA_VISIBLE_DEVICES=$GPU_ID python3 ./src/train_emar.py --config_path "$CONFIG_PATH" &

  # ★ ここで1分待機してから次のジョブを投げる
  sleep 60
done

# すべてのジョブが完了するのを待つ
wait

echo "All jobs finished."

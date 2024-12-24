export CUDA_VISIBLE_DEVICES=0
python ./src/sampling.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --output_dir ./results/ad_dit_d8w768/hazelnut/ \
    --model_ckpt ./results/ad_dit_d8w768/hazelnut/model_ema_latest.pth \
    --config_path ./results/ad_dit_d8w768/hazelnut/config.yaml \
    --device cuda
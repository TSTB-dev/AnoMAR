export CUDA_VISIBLE_DEVICES=1
python ./src/sampling.py \
    --num_samples 1 \
    --num_inference_steps 10 \
    --output_dir /home/haselab/projects/sakai/AnoMAR/ad_dit_d8w768/ \
    --model_ckpt /home/haselab/projects/sakai/AnoMAR/ad_dit_d8w768/model_ema_latest.pth \
    --config_path /home/haselab/projects/sakai/AnoMAR/ad_dit_d8w768/config.yaml \
    --device cuda
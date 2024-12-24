export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space latent \
    --start_step 10 \
    --output_dir ./results/ad_dit_d8w768/bottle \
    --model_ckpt ./results/ad_dit_d8w768/bottle/model_latest.pth \
    --config_path ./results/ad_dit_d8w768/bottle/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 4 \
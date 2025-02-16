export CUDA_VISIBLE_DEVICES=5
python src/train_da.py \
    --num_epochs 1 \
    --batch_size 8 \
    --lr 0.0001 \
    --diffusion_config /home/haselab/projects/sakai/AnoMAR/AnoMAR/results/ad_unet_vae_rot/leather/config.yaml \
    --diffusion_ckpt /home/haselab/projects/sakai/AnoMAR/AnoMAR/ddad_models/leather \
    --backbone_name unet \
    --mask_strategy checkerboard \
    --num_inference_steps 50 \
    --num_samples 1 \
    --save_dir ./results_da \
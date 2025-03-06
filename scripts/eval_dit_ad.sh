# # bottle  cable  capsule  carpet  grid  hazelnut  leather  metal_nut  pill  bottle  tile  toothbrush  transistor  wood  zipper

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_dit.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_unet_vae_rot/bottle \
#     --model_ckpt ./bottle \
#     --config_path ./results/ad_unet_vae_rot/bottle/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_dit.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_unet_vae_rot/cable \
#     --model_ckpt ./cable \
#     --config_path ./results/ad_unet_vae_rot/cable/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_dit.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 8 \
#     --output_dir ./results/exp_dit_vae_ad/capsule \
#     --model_ckpt ./results/exp_dit_vae_ad/capsule/model_ema_latest.pth \
#     --config_path ./results/exp_dit_vae_ad/capsule/config.yaml \
#     --save_images \
#     --save_all_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_dit.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_unet_vae_rot/carpet \
#     --model_ckpt ./carpet \
#     --config_path ./results/ad_unet_vae_rot/carpet/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_dit.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 8 \
#     --output_dir ./results/exp_dit_vae_ad/grid \
#     --model_ckpt ./results/exp_dit_vae_ad/grid/model_ema_latest.pth \
#     --config_path ./results/exp_dit_vae_ad/grid/config.yaml \
#     --save_images \
#     --save_all_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_dit.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_unet_vae_rot/hazelnut \
#     --model_ckpt ./hazelnut \
#     --config_path ./results/ad_unet_vae_rot/hazelnut/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_dit.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 32 \
#     --output_dir ./results/exp_dit_vae_ad/leather \
#     --model_ckpt ./results/exp_dit_vae_ad/leather/model_ema_latest.pth \
#     --config_path ./results/exp_dit_vae_ad/leather/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_dit.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_unet_vae_rot/metal_nut \
#     --model_ckpt ./metal_nut \
#     --config_path ./results/ad_unet_vae_rot/metal_nut/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_dit.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_unet_vae_rot/pill \
#     --model_ckpt ./pill \
#     --config_path ./results/ad_unet_vae_rot/pill/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1


# ==============================================================================
# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 8 \
#     --output_dir ./results/w2d256ep1000/ \
#     --model_ckpt ./results/w2d256ep1000/model_latest.pth \
#     --config_path ./results/w2d256ep1000/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --eval_dataset mvtec_ad \
#     --eval_category tile \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/w2d256ep1000/ \
#     --model_ckpt ./results/w2d256ep1000/model_latest.pth \
#     --config_path ./results/w2d256ep1000/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --eval_dataset mvtec_ad \
#     --eval_category tile \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 32 \
#     --output_dir ./results/w2d256ep1000/ \
#     --model_ckpt ./results/w2d256ep1000/model_latest.pth \
#     --config_path ./results/w2d256ep1000/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --eval_dataset mvtec_ad \
#     --eval_category tile \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 8 \
#     --output_dir ./results/w2d256ep1000/ \
#     --model_ckpt ./results/w2d256ep1000/model_latest.pth \
#     --config_path ./results/w2d256ep1000/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --eval_dataset mvtec_ad \
#     --eval_category toothbrush \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/w2d256ep1000/ \
#     --model_ckpt ./results/w2d256ep1000/model_latest.pth \
#     --config_path ./results/w2d256ep1000/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --eval_dataset mvtec_ad \
#     --eval_category toothbrush \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 32 \
#     --output_dir ./results/w2d256ep1000/ \
#     --model_ckpt ./results/w2d256ep1000/model_latest.pth \
#     --config_path ./results/w2d256ep1000/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --eval_dataset mvtec_ad \
#     --eval_category toothbrush \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 8 \
#     --output_dir ./results/w2d256ep1000/ \
#     --model_ckpt ./results/w2d256ep1000/model_latest.pth \
#     --config_path ./results/w2d256ep1000/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --eval_dataset mvtec_ad \
#     --eval_category transistor \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/w2d256ep1000/ \
#     --model_ckpt ./results/w2d256ep1000/model_latest.pth \
#     --config_path ./results/w2d256ep1000/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --eval_dataset mvtec_ad \
#     --eval_category transistor \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 32 \
#     --output_dir ./results/w2d256ep1000/ \
#     --model_ckpt ./results/w2d256ep1000/model_latest.pth \
#     --config_path ./results/w2d256ep1000/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --eval_dataset mvtec_ad \
#     --eval_category transistor \

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit_feature.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 8 \
    --output_dir ./results/exp_dit_feature_d4w256_lr/wood \
    --model_ckpt ./results/exp_dit_feature_d4w256_lr/wood/model_latest.pth \
    --config_path ./results/exp_dit_feature_d4w256_lr/wood/config.yaml \
    --device cuda \
    --batch_size 1 \

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit_feature.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 16 \
    --output_dir ./results/exp_dit_feature_d4w256_lr/wood \
    --model_ckpt ./results/exp_dit_feature_d4w256_lr/wood/model_latest.pth \
    --config_path ./results/exp_dit_feature_d4w256_lr/wood/config.yaml \
    --device cuda \
    --batch_size 1 \

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit_feature.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 32 \
    --output_dir ./results/exp_dit_feature_d4w256_lr/wood \
    --model_ckpt ./results/exp_dit_feature_d4w256_lr/wood/model_latest.pth \
    --config_path ./results/exp_dit_feature_d4w256_lr/wood/config.yaml \
    --device cuda \
    --batch_size 1 \

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit_feature.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 8 \
    --output_dir ./results/exp_dit_feature_d4w256_lr/zipper \
    --model_ckpt ./results/exp_dit_feature_d4w256_lr/zipper/model_latest.pth \
    --config_path ./results/exp_dit_feature_d4w256_lr/zipper/config.yaml \
    --device cuda \
    --batch_size 1 \

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit_feature.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 16 \
    --output_dir ./results/exp_dit_feature_d4w256_lr/zipper \
    --model_ckpt ./results/exp_dit_feature_d4w256_lr/zipper/model_latest.pth \
    --config_path ./results/exp_dit_feature_d4w256_lr/zipper/config.yaml \
    --device cuda \
    --batch_size 1 \

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit_feature.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 32 \
    --output_dir ./results/exp_dit_feature_d4w256_lr/zipper \
    --model_ckpt ./results/exp_dit_feature_d4w256_lr/zipper/model_latest.pth \
    --config_path ./results/exp_dit_feature_d4w256_lr/zipper/config.yaml \
    --device cuda \
    --batch_size 1 \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 8 \
#     --output_dir ./results/exp_dit_feature_d4w256_lr/toothbrush \
#     --model_ckpt ./results/exp_dit_feature_d4w256_lr/toothbrush/model_latest.pth \
#     --config_path ./results/exp_dit_feature_d4w256_lr/toothbrush/config.yaml \
#     --device cuda \
#     --batch_size 1 \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/exp_dit_feature_d4w256_lr/toothbrush \
#     --model_ckpt ./results/exp_dit_feature_d4w256_lr/toothbrush/model_latest.pth \
#     --config_path ./results/exp_dit_feature_d4w256_lr/toothbrush/config.yaml \
#     --device cuda \
#     --batch_size 1 \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 32 \
#     --output_dir ./results/exp_dit_feature_d4w256_lr/toothbrush \
#     --model_ckpt ./results/exp_dit_feature_d4w256_lr/toothbrush/model_latest.pth \
#     --config_path ./results/exp_dit_feature_d4w256_lr/toothbrush/config.yaml \
#     --device cuda \
#     --batch_size 1 \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 8 \
#     --output_dir ./results/exp_dit_feature_d4w256_lr/transistor \
#     --model_ckpt ./results/exp_dit_feature_d4w256_lr/transistor/model_latest.pth \
#     --config_path ./results/exp_dit_feature_d4w256_lr/transistor/config.yaml \
#     --device cuda \
#     --batch_size 1 \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/exp_dit_feature_d4w256_lr/transistor \
#     --model_ckpt ./results/exp_dit_feature_d4w256_lr/transistor/model_latest.pth \
#     --config_path ./results/exp_dit_feature_d4w256_lr/transistor/config.yaml \
#     --device cuda \
#     --batch_size 1 \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 32 \
#     --output_dir ./results/exp_dit_feature_d4w256_lr/transistor \
#     --model_ckpt ./results/exp_dit_feature_d4w256_lr/transistor/model_latest.pth \
#     --config_path ./results/exp_dit_feature_d4w256_lr/transistor/config.yaml \
#     --device cuda \
#     --batch_size 1 \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 8 \
#     --output_dir ./results/exp_dit_feature_d4w256/cable \
#     --model_ckpt ./results/exp_dit_feature_d4w256/cable/model_ema_latest.pth \
#     --config_path ./results/exp_dit_feature_d4w256/cable/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --save_all_images \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 8 \
#     --output_dir ./results/exp_dit_feature_d4w256/capsule \
#     --model_ckpt ./results/exp_dit_feature_d4w256/capsule/model_ema_latest.pth \
#     --config_path ./results/exp_dit_feature_d4w256/capsule/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --save_all_images \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 8 \
#     --output_dir ./results/exp_dit_feature_d4w256/carpet \
#     --model_ckpt ./results/exp_dit_feature_d4w256/carpet/model_ema_latest.pth \
#     --config_path ./results/exp_dit_feature_d4w256/carpet/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --save_all_images \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 8 \
#     --output_dir ./results/exp_dit_feature_d4w256/grid \
#     --model_ckpt ./results/exp_dit_feature_d4w256/grid/model_ema_latest.pth \
#     --config_path ./results/exp_dit_feature_d4w256/grid/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --save_all_images \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 8 \
#     --output_dir ./results/exp_dit_feature_d4w256/hazelnut \
#     --model_ckpt ./results/exp_dit_feature_d4w256/hazelnut/model_ema_latest.pth \
#     --config_path ./results/exp_dit_feature_d4w256/hazelnut/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --save_all_images \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 8 \
#     --output_dir ./results/exp_dit_feature_d4w256/leather \
#     --model_ckpt ./results/exp_dit_feature_d4w256/leather/model_ema_latest.pth \
#     --config_path ./results/exp_dit_feature_d4w256/leather/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --save_all_images \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 8 \
#     --output_dir ./results/exp_dit_feature_d4w256/metal_nut \
#     --model_ckpt ./results/exp_dit_feature_d4w256/metal_nut/model_ema_latest.pth \
#     --config_path ./results/exp_dit_feature_d4w256/metal_nut/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --save_all_images \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 8 \
#     --output_dir ./results/exp_dit_feature_d4w256/pill \
#     --model_ckpt ./results/exp_dit_feature_d4w256/pill/model_ema_latest.pth \
#     --config_path ./results/exp_dit_feature_d4w256/pill/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --save_all_images \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 8 \
#     --output_dir ./results/exp_dit_feature_d4w256/screw \
#     --model_ckpt ./results/exp_dit_feature_d4w256/screw/model_ema_latest.pth \
#     --config_path ./results/exp_dit_feature_d4w256/screw/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --save_all_images \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 8 \
#     --output_dir ./results/exp_dit_feature_d4w256/tile \
#     --model_ckpt ./results/exp_dit_feature_d4w256/tile/model_ema_latest.pth \
#     --config_path ./results/exp_dit_feature_d4w256/tile/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --save_all_images \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 8 \
#     --output_dir ./results/exp_dit_feature_d4w256/toothbrush \
#     --model_ckpt ./results/exp_dit_feature_d4w256/toothbrush/model_ema_latest.pth \
#     --config_path ./results/exp_dit_feature_d4w256/toothbrush/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --save_all_images \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 8 \
#     --output_dir ./results/exp_dit_feature_d4w256/transistor \
#     --model_ckpt ./results/exp_dit_feature_d4w256/transistor/model_ema_latest.pth \
#     --config_path ./results/exp_dit_feature_d4w256/transistor/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --save_all_images \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 8 \
#     --output_dir ./results/exp_dit_feature_d4w256/wood \
#     --model_ckpt ./results/exp_dit_feature_d4w256/wood/model_ema_latest.pth \
#     --config_path ./results/exp_dit_feature_d4w256/wood/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --save_all_images \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 8 \
#     --output_dir ./results/exp_dit_feature_d4w256/zipper \
#     --model_ckpt ./results/exp_dit_feature_d4w256/zipper/model_ema_latest.pth \
#     --config_path ./results/exp_dit_feature_d4w256/zipper/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --save_all_images \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/exp_unet_feature_ad/bottle \
#     --model_ckpt ./results/exp_unet_feature_ad/bottle/model_ema_latest.pth \
#     --config_path ./results/exp_unet_feature_ad/bottle/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --save_all_images \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 32 \
#     --output_dir ./results/exp_unet_feature_ad/bottle \
#     --model_ckpt ./results/exp_unet_feature_ad/bottle/model_ema_latest.pth \
#     --config_path ./results/exp_unet_feature_ad/bottle/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --save_all_images \

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit_feature.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 48 \
#     --output_dir ./results/exp_unet_vae_ad/grid \
#     --model_ckpt ./results/exp_unet_vae_ad/grid/model_ema_latest.pth \
#     --config_path ./results/exp_unet_vae_ad/grid/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --save_all_images \

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_dit.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_unet_vae_rot/tile \
#     --model_ckpt ./tile \
#     --config_path ./results/ad_unet_vae_rot/tile/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_dit.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_unet_vae_rot/toothbrush \
#     --model_ckpt ./toothbrush \
#     --config_path ./results/ad_unet_vae_rot/toothbrush/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_dit.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_unet_vae_rot/transistor \
#     --model_ckpt ./transistor \
#     --config_path ./results/ad_unet_vae_rot/transistor/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_dit.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_unet_vae_rot/wood \
#     --model_ckpt ./wood \
#     --config_path ./results/ad_unet_vae_rot/wood/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_dit.py \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_unet_vae_rot/zipper \
#     --model_ckpt ./zipper \
#     --config_path ./results/ad_unet_vae_rot/zipper/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# ====================

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit.py \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --recon_space latent \
#     --start_step 16 \
#     --output_dir ./results/loco_dit_vae/breakfast_box \
#     --model_ckpt ./results/loco_dit_vae/breakfast_box/model_ema_latest.pth \
#     --config_path ./results/loco_dit_vae/breakfast_box/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit.py \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --recon_space latent \
#     --start_step 16 \
#     --output_dir ./results/loco_dit_vae/pushpins \
#     --model_ckpt ./results/loco_dit_vae/pushpins/model_ema_latest.pth \
#     --config_path ./results/loco_dit_vae/pushpins/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit.py \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --recon_space latent \
#     --start_step 16 \
#     --output_dir ./results/loco_dit_vae/bottle_bag \
#     --model_ckpt ./results/loco_dit_vae/bottle_bag/model_ema_latest.pth \
#     --config_path ./results/loco_dit_vae/bottle_bag/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit.py \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --recon_space latent \
#     --start_step 16 \
#     --output_dir ./results/loco_dit_vae/splicing_connectors \
#     --model_ckpt ./results/loco_dit_vae/splicing_connectors/model_ema_latest.pth \
#     --config_path ./results/loco_dit_vae/splicing_connectors/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/cable \
#     --model_ckpt ./results/ad_dit_d8w768/cable/model_ema_latest.pth \
#     --config_path ./results/ad_dit_d8w768/cable/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/grid \
#     --model_ckpt ./results/ad_dit_d8w768/grid/model_ema_latest.pth \
#     --config_path ./results/ad_dit_d8w768/capsule/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/carpet \
#     --model_ckpt ./results/ad_dit_d8w768/carpet/model_ema_latest.pth \
#     --config_path ./results/ad_dit_d8w768/carpet/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/grid \
#     --model_ckpt ./results/ad_dit_d8w768/grid/model_ema_latest.pth \
#     --config_path ./results/ad_dit_d8w768/grid/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/hazelnut \
#     --model_ckpt ./results/ad_dit_d8w768/hazelnut/model_ema_latest.pth \
#     --config_path ./results/ad_dit_d8w768/hazelnut/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/leather \
#     --model_ckpt ./results/ad_dit_d8w768/leather/model_ema_latest.pth \
#     --config_path ./results/ad_dit_d8w768/leather/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/metal_nut \
#     --model_ckpt ./results/ad_dit_d8w768/metal_nut/model_ema_latest.pth \
#     --config_path ./results/ad_dit_d8w768/metal_nut/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/pill \
#     --model_ckpt ./results/ad_dit_d8w768/pill/model_ema_latest.pth \
#     --config_path ./results/ad_dit_d8w768/pill/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/bottle \
#     --model_ckpt ./results/ad_dit_d8w768/bottle/model_ema_latest.pth \
#     --config_path ./results/ad_dit_d8w768/bottle/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/tile \
#     --model_ckpt ./results/ad_dit_d8w768/tile/model_ema_latest.pth \
#     --config_path ./results/ad_dit_d8w768/tile/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/toothbrush \
#     --model_ckpt ./results/ad_dit_d8w768/toothbrush/model_ema_latest.pth \
#     --config_path ./results/ad_dit_d8w768/toothbrush/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/transistor \
#     --model_ckpt ./results/ad_dit_d8w768/transistor/model_ema_latest.pth \
#     --config_path ./results/ad_dit_d8w768/transistor/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/wood \
#     --model_ckpt ./results/ad_dit_d8w768/wood/model_ema_latest.pth \
#     --config_path ./results/ad_dit_d8w768/wood/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_dit.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/zipper \
#     --model_ckpt ./results/ad_dit_d8w768/zipper/model_ema_latest.pth \
#     --config_path ./results/ad_dit_d8w768/zipper/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

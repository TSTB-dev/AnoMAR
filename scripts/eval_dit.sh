# # bottle  cable  capsule  carpet  grid  hazelnut  leather  metal_nut  pill  bottle  tile  toothbrush  transistor  wood  zipper

export CUDA_VISIBLE_DEVICES=4
python src/evaluate_ddad.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 16 \
    --output_dir ./results/ad_unet_vae_rot/bottle \
    --model_ckpt ./bottle \
    --config_path ./results/ad_unet_vae_rot/bottle/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=4
python src/evaluate_ddad.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 16 \
    --output_dir ./results/ad_unet_vae_rot/cable \
    --model_ckpt ./cable \
    --config_path ./results/ad_unet_vae_rot/cable/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=4
python src/evaluate_ddad.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 16 \
    --output_dir ./results/ad_unet_vae_latent/capsule \
    --model_ckpt ./capsule \
    --config_path ./results/ad_unet_vae_latent/capsule/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=4
python src/evaluate_ddad.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 16 \
    --output_dir ./results/ad_unet_vae_rot/carpet \
    --model_ckpt ./carpet \
    --config_path ./results/ad_unet_vae_rot/carpet/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=4
python src/evaluate_ddad.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 16 \
    --output_dir ./results/ad_unet_vae_rot/grid \
    --model_ckpt ./grid \
    --config_path ./results/ad_unet_vae_rot/grid/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=4
python src/evaluate_ddad.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 16 \
    --output_dir ./results/ad_unet_vae_rot/hazelnut \
    --model_ckpt ./hazelnut \
    --config_path ./results/ad_unet_vae_rot/hazelnut/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=4
python src/evaluate_ddad.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 16 \
    --output_dir ./results/ad_unet_vae_rot/leather \
    --model_ckpt ./leather \
    --config_path ./results/ad_unet_vae_rot/leather/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=4
python src/evaluate_ddad.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 16 \
    --output_dir ./results/ad_unet_vae_rot/metal_nut \
    --model_ckpt ./metal_nut \
    --config_path ./results/ad_unet_vae_rot/metal_nut/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=4
python src/evaluate_ddad.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 16 \
    --output_dir ./results/ad_unet_vae_rot/pill \
    --model_ckpt ./pill \
    --config_path ./results/ad_unet_vae_rot/pill/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=4
python src/evaluate_ddad.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 16 \
    --output_dir ./results/ad_unet_vae_rot/screw \
    --model_ckpt ./screw \
    --config_path ./results/ad_unet_vae_rot/screw/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=4
python src/evaluate_ddad.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 16 \
    --output_dir ./results/ad_unet_vae_rot/tile \
    --model_ckpt ./tile \
    --config_path ./results/ad_unet_vae_rot/tile/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=4
python src/evaluate_ddad.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 16 \
    --output_dir ./results/ad_unet_vae_rot/toothbrush \
    --model_ckpt ./toothbrush \
    --config_path ./results/ad_unet_vae_rot/toothbrush/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=4
python src/evaluate_ddad.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 16 \
    --output_dir ./results/ad_unet_vae_rot/transistor \
    --model_ckpt ./transistor \
    --config_path ./results/ad_unet_vae_rot/transistor/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=4
python src/evaluate_ddad.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 16 \
    --output_dir ./results/ad_unet_vae_rot/wood \
    --model_ckpt ./wood \
    --config_path ./results/ad_unet_vae_rot/wood/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=4
python src/evaluate_ddad.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 16 \
    --output_dir ./results/ad_unet_vae_rot/zipper \
    --model_ckpt ./zipper \
    --config_path ./results/ad_unet_vae_rot/zipper/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

# ====================

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_ddad.py \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --recon_space latent \
#     --start_step 16 \
#     --output_dir ./results/loco_dit_vae/breakfast_box \
#     --model_ckpt ./results/loco_dit_vae/breakfast_box/model_latest.pth \
#     --config_path ./results/loco_dit_vae/breakfast_box/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_ddad.py \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --recon_space latent \
#     --start_step 16 \
#     --output_dir ./results/loco_dit_vae/pushpins \
#     --model_ckpt ./results/loco_dit_vae/pushpins/model_latest.pth \
#     --config_path ./results/loco_dit_vae/pushpins/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_ddad.py \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --recon_space latent \
#     --start_step 16 \
#     --output_dir ./results/loco_dit_vae/bottle_bag \
#     --model_ckpt ./results/loco_dit_vae/bottle_bag/model_latest.pth \
#     --config_path ./results/loco_dit_vae/bottle_bag/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_ddad.py \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --recon_space latent \
#     --start_step 16 \
#     --output_dir ./results/loco_dit_vae/splicing_connectors \
#     --model_ckpt ./results/loco_dit_vae/splicing_connectors/model_latest.pth \
#     --config_path ./results/loco_dit_vae/splicing_connectors/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_ddad.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/cable \
#     --model_ckpt ./results/ad_dit_d8w768/cable/model_latest.pth \
#     --config_path ./results/ad_dit_d8w768/cable/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_ddad.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/capsule \
#     --model_ckpt ./results/ad_dit_d8w768/capsule/model_latest.pth \
#     --config_path ./results/ad_dit_d8w768/capsule/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_ddad.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/carpet \
#     --model_ckpt ./results/ad_dit_d8w768/carpet/model_latest.pth \
#     --config_path ./results/ad_dit_d8w768/carpet/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_ddad.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/grid \
#     --model_ckpt ./results/ad_dit_d8w768/grid/model_latest.pth \
#     --config_path ./results/ad_dit_d8w768/grid/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_ddad.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/hazelnut \
#     --model_ckpt ./results/ad_dit_d8w768/hazelnut/model_latest.pth \
#     --config_path ./results/ad_dit_d8w768/hazelnut/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_ddad.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/leather \
#     --model_ckpt ./results/ad_dit_d8w768/leather/model_latest.pth \
#     --config_path ./results/ad_dit_d8w768/leather/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_ddad.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/metal_nut \
#     --model_ckpt ./results/ad_dit_d8w768/metal_nut/model_latest.pth \
#     --config_path ./results/ad_dit_d8w768/metal_nut/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_ddad.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/pill \
#     --model_ckpt ./results/ad_dit_d8w768/pill/model_latest.pth \
#     --config_path ./results/ad_dit_d8w768/pill/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_ddad.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/bottle \
#     --model_ckpt ./results/ad_dit_d8w768/bottle/model_latest.pth \
#     --config_path ./results/ad_dit_d8w768/bottle/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_ddad.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/tile \
#     --model_ckpt ./results/ad_dit_d8w768/tile/model_latest.pth \
#     --config_path ./results/ad_dit_d8w768/tile/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_ddad.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/toothbrush \
#     --model_ckpt ./results/ad_dit_d8w768/toothbrush/model_latest.pth \
#     --config_path ./results/ad_dit_d8w768/toothbrush/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_ddad.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/transistor \
#     --model_ckpt ./results/ad_dit_d8w768/transistor/model_latest.pth \
#     --config_path ./results/ad_dit_d8w768/transistor/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_ddad.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/wood \
#     --model_ckpt ./results/ad_dit_d8w768/wood/model_latest.pth \
#     --config_path ./results/ad_dit_d8w768/wood/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_ddad.py \
#     --num_samples 32 \
#     --num_inference_steps 100 \
#     --recon_space feature \
#     --start_step 16 \
#     --output_dir ./results/ad_dit_d8w768/zipper \
#     --model_ckpt ./results/ad_dit_d8w768/zipper/model_latest.pth \
#     --config_path ./results/ad_dit_d8w768/zipper/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 1

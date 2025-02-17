# # # bottle  cable  capsule  carpet  grid  hazelnut  leather  metal_nut  pill  bottle  tile  toothbrush  transistor  wood  zipper

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_ddrm.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --num_iterations 1 \
#     --quant_thresh 0.8 \
#     --recon_space feature \
#     --aggregation mean \
#     --eta 0.85 \
#     --etaB 1.0 \
#     --output_dir ./results/ad_unet_vae_rot/bottle \
#     --model_ckpt ./ddad_models/bottle \
#     --config_path ./results/ad_unet_vae_rot/bottle/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 8 \
#     --sample_indices 0 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_ddrm.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --num_iterations 1 \
#     --quant_thresh 0.8 \
#     --recon_space feature \
#     --aggregation mean \
#     --eta 0.85 \
#     --etaB 1.0 \
#     --output_dir ./results/ad_unet_vae_rot/cable \
#     --model_ckpt ./ddad_models/cable \
#     --config_path ./results/ad_unet_vae_rot/cable/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 8 \
#     --sample_indices 0 1

export CUDA_VISIBLE_DEVICES=4
python src/evaluate_ddrm.py \
    --num_masks 1 \
    --num_samples 1 \
    --num_inference_steps 100 \
    --num_iterations 1 \
    --quant_thresh 0.8 \
    --recon_space feature \
    --aggregation mean \
    --eta 0.85 \
    --etaB 1.0 \
    --output_dir ./results/exp_dit_vae_ad/capsule \
    --model_ckpt ./results/exp_dit_vae_ad/capsule/model_latest.pth \
    --config_path ./results/exp_dit_vae_ad/capsule/config.yaml \
    --device cuda \
    --save_images \
    --batch_size 8 \
    --sample_indices 0 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_ddrm.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --num_iterations 1 \
#     --quant_thresh 0.8 \
#     --recon_space feature \
#     --aggregation mean \
#     --eta 0.85 \
#     --etaB 1.0 \
#     --output_dir ./results/ad_unet_vae_rot/carpet \
#     --model_ckpt ./ddad_models/carpet \
#     --config_path ./results/ad_unet_vae_rot/carpet/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 8 \
#     --sample_indices 0 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_ddrm.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --num_iterations 1 \
#     --quant_thresh 0.8 \
#     --recon_space feature \
#     --aggregation mean \
#     --eta 0.85 \
#     --etaB 1.0 \
#     --output_dir ./results/ad_unet_vae_rot/grid \
#     --model_ckpt ./ddad_models/grid \
#     --config_path ./results/ad_unet_vae_rot/grid/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 8 \
#     --sample_indices 0 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_ddrm.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --num_iterations 1 \
#     --quant_thresh 0.8 \
#     --recon_space feature \
#     --aggregation mean \
#     --eta 0.85 \
#     --etaB 1.0 \
#     --output_dir ./results/ad_unet_vae_rot/hazelnut \
#     --model_ckpt ./ddad_models/hazelnut \
#     --config_path ./results/ad_unet_vae_rot/hazelnut/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 8 \
#     --sample_indices 0 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_ddrm.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --num_iterations 1 \
#     --quant_thresh 0.8 \
#     --recon_space feature \
#     --aggregation mean \
#     --eta 0.85 \
#     --etaB 1.0 \
#     --output_dir ./results/ad_unet_vae_rot/leather \
#     --model_ckpt ./ddad_models/leather \
#     --config_path ./results/ad_unet_vae_rot/leather/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 8 \
#     --sample_indices 0 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_ddrm.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --num_iterations 1 \
#     --quant_thresh 0.8 \
#     --recon_space feature \
#     --aggregation mean \
#     --eta 0.85 \
#     --etaB 1.0 \
#     --output_dir ./results/ad_unet_vae_rot/metal_nut \
#     --model_ckpt ./ddad_models/metal_nut \
#     --config_path ./results/ad_unet_vae_rot/metal_nut/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 8 \
#     --sample_indices 0 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_ddrm.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --num_iterations 1 \
#     --quant_thresh 0.8 \
#     --recon_space feature \
#     --aggregation mean \
#     --eta 0.85 \
#     --etaB 1.0 \
#     --output_dir ./results/ad_unet_vae_rot/pill \
#     --model_ckpt ./ddad_models/pill \
#     --config_path ./results/ad_unet_vae_rot/pill/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 8 \
#     --sample_indices 0 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_ddrm.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --num_iterations 1 \
#     --quant_thresh 0.8 \
#     --recon_space feature \
#     --aggregation mean \
#     --eta 0.85 \
#     --etaB 1.0 \
#     --output_dir ./results/ad_unet_vae_rot/screw \
#     --model_ckpt ./ddad_models/screw \
#     --config_path ./results/ad_unet_vae_rot/screw/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 8 \
#     --sample_indices 0 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_ddrm.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --num_iterations 1 \
#     --quant_thresh 0.8 \
#     --recon_space feature \
#     --aggregation mean \
#     --eta 0.85 \
#     --etaB 1.0 \
#     --output_dir ./results/ad_unet_vae_rot/tile \
#     --model_ckpt ./ddad_models/tile \
#     --config_path ./results/ad_unet_vae_rot/tile/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 30 \
#     --sample_indices 0 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_ddrm.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --num_iterations 1 \
#     --quant_thresh 0.8 \
#     --recon_space feature \
#     --aggregation mean \
#     --eta 0.85 \
#     --etaB 1.0 \
#     --output_dir ./results/ad_unet_vae_rot/toothbrush \
#     --model_ckpt ./ddad_models/toothbrush \
#     --config_path ./results/ad_unet_vae_rot/toothbrush/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 8 \
#     --sample_indices 0 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_ddrm.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --num_iterations 1 \
#     --quant_thresh 0.8 \
#     --recon_space feature \
#     --aggregation mean \
#     --eta 0.85 \
#     --etaB 1.0 \
#     --output_dir ./results/ad_unet_vae_rot/transistor \
#     --model_ckpt ./ddad_models/transistor \
#     --config_path ./results/ad_unet_vae_rot/transistor/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 8 \
#     --sample_indices 0 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_ddrm.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --num_iterations 1 \
#     --quant_thresh 0.8 \
#     --recon_space feature \
#     --aggregation mean \
#     --eta 0.85 \
#     --etaB 1.0 \
#     --output_dir ./results/ad_unet_vae_rot/wood \
#     --model_ckpt ./ddad_models/wood \
#     --config_path ./results/ad_unet_vae_rot/wood/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 8 \
#     --sample_indices 0 1

# export CUDA_VISIBLE_DEVICES=4
# python src/evaluate_ddrm.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 100 \
#     --num_iterations 1 \
#     --quant_thresh 0.8 \
#     --recon_space feature \
#     --aggregation mean \
#     --eta 0.85 \
#     --etaB 1.0 \
#     --output_dir ./results/ad_unet_vae_rot/zipper \
#     --model_ckpt ./ddad_models/zipper \
#     --config_path ./results/ad_unet_vae_rot/zipper/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 8 \
#     --sample_indices 0 1

# ============

# export CUDA_VISIBLE_DEVICES=1
# python src/evaluate_repaint.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 50 \
#    --start_step 64 \
#     --recon_space latent \
#     --aggregation mean \
#     --num_resamples 4 \
#     --num_jumps 4 \
#     --output_dir ./results/loco_dit_vae/juice_bottle \
#     --model_ckpt ./results/loco_dit_vae/juice_bottle/model_latest.pth \
#     --config_path ./results/loco_dit_vae/juice_bottle/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 8 \
#     --sample_indices 10 20

# export CUDA_VISIBLE_DEVICES=1
# python src/evaluate_repaint.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 50 \
#    --start_step 64 \
#     --recon_space latent \
#     --aggregation mean \
#     --num_resamples 4 \
#     --num_jumps 4 \
#     --output_dir ./results/loco_dit_vae/pushpins \
#     --model_ckpt ./results/loco_dit_vae/pushpins/model_latest.pth \
#     --config_path ./results/loco_dit_vae/pushpins/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 8 \
#     --sample_indices 10 20

# export CUDA_VISIBLE_DEVICES=1
# python src/evaluate_repaint.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 50 \
#    --start_step 64 \
#     --recon_space latent \
#     --aggregation mean \
#     --num_resamples 4 \
#     --num_jumps 4 \
#     --output_dir ./results/loco_dit_vae/bottle_bag \
#     --model_ckpt ./results/loco_dit_vae/bottle_bag/model_latest.pth \
#     --config_path ./results/loco_dit_vae/bottle_bag/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 8 \
#     --sample_indices 10 20

# export CUDA_VISIBLE_DEVICES=1
# python src/evaluate_repaint.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 50 \
#    --start_step 64 \
#     --recon_space latent \
#     --aggregation mean \
#     --num_resamples 4 \
#     --num_jumps 4 \
#     --output_dir ./results/loco_dit_vae/splicing_connectors \
#     --model_ckpt ./results/loco_dit_vae/splicing_connectors/model_latest.pth \
#     --config_path ./results/loco_dit_vae/splicing_connectors/config.yaml \
#     --save_images \
#     --device cuda \
#     --batch_size 8 \
#     --sample_indices 10 20

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 50 \
#    --start_step 64 \
#    --sampler ddim \
#    --eta 0.0 \
#    --temperature 1 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_ca_base_enet/bottle \
#     --model_ckpt ./results/ad_mar_ca_base_enet/bottle/model_latest.pth \
#     --config_path ./results/ad_mar_ca_base_enet/bottle/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 50 \
#    --start_step 64 \
#    --sampler ddim \
#    --eta 0.0 \
#    --temperature 10 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_ca_base_enet/cable \
#     --model_ckpt ./results/ad_mar_ca_base_enet/cable/model_latest.pth \
#     --config_path ./results/ad_mar_ca_base_enet/cable/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 50 \
#    --start_step 64 \
#    --sampler ddim \
#    --eta 0.0 \
#    --temperature 10 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_ca_base_enet/capsule \
#     --model_ckpt ./results/ad_mar_ca_base_enet/capsule/model_latest.pth \
#     --config_path ./results/ad_mar_ca_base_enet/capsule/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 50 \
#    --start_step 64 \
#    --sampler ddim \
#    --eta 0.0 \
#    --temperature 10 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_ca_base_enet/carpet \
#     --model_ckpt ./results/ad_mar_ca_base_enet/carpet/model_latest.pth \
#     --config_path ./results/ad_mar_ca_base_enet/carpet/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 50 \
#    --start_step 64 \
#    --sampler ddim \
#    --eta 0.0 \
#    --temperature 10 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_ca_base_enet/grid \
#     --model_ckpt ./results/ad_mar_ca_base_enet/grid/model_latest.pth \
#     --config_path ./results/ad_mar_ca_base_enet/grid/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 50 \
#    --start_step 64 \
#    --sampler ddim \
#    --eta 0.0 \
#    --temperature 10 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_ca_base_enet/hazelnut \
#     --model_ckpt ./results/ad_mar_ca_base_enet/hazelnut/model_latest.pth \
#     --config_path ./results/ad_mar_ca_base_enet/hazelnut/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 50 \
#    --start_step 64 \
#    --sampler ddim \
#    --eta 0.0 \
#    --temperature 10 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_ca_base_enet/leather \
#     --model_ckpt ./results/ad_mar_ca_base_enet/leather/model_latest.pth \
#     --config_path ./results/ad_mar_ca_base_enet/leather/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 50 \
#    --start_step 64 \
#    --sampler ddim \
#    --eta 0.0 \
#    --temperature 10 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_ca_base_enet/metal_nut \
#     --model_ckpt ./results/ad_mar_ca_base_enet/metal_nut/model_latest.pth \
#     --config_path ./results/ad_mar_ca_base_enet/metal_nut/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 50 \
#    --start_step 64 \
#    --sampler ddim \
#    --eta 0.0 \
#    --temperature 10 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_ca_base_enet/pill \
#     --model_ckpt ./results/ad_mar_ca_base_enet/pill/model_latest.pth \
#     --config_path ./results/ad_mar_ca_base_enet/pill/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 50 \
#    --start_step 64 \
#    --sampler ddim \
#    --eta 0.0 \
#    --temperature 10 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_ca_base_enet/bottle \
#     --model_ckpt ./results/ad_mar_ca_base_enet/bottle/model_latest.pth \
#     --config_path ./results/ad_mar_ca_base_enet/bottle/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 50 \
#    --start_step 64 \
#    --sampler ddim \
#    --eta 0.0 \
#    --temperature 10 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_ca_base_enet/tile \
#     --model_ckpt ./results/ad_mar_ca_base_enet/tile/model_latest.pth \
#     --config_path ./results/ad_mar_ca_base_enet/tile/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 50 \
#    --start_step 64 \
#    --sampler ddim \
#    --eta 0.0 \
#    --temperature 10 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_ca_base_enet/toothbrush \
#     --model_ckpt ./results/ad_mar_ca_base_enet/toothbrush/model_latest.pth \
#     --config_path ./results/ad_mar_ca_base_enet/toothbrush/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 50 \
#    --start_step 64 \
#    --sampler ddim \
#    --eta 0.0 \
#    --temperature 10 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_ca_base_enet/transistor \
#     --model_ckpt ./results/ad_mar_ca_base_enet/transistor/model_latest.pth \
#     --config_path ./results/ad_mar_ca_base_enet/transistor/config.yaml

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 50 \
#    --start_step 64 \
#    --sampler ddim \
#    --eta 0.0 \
#    --temperature 10 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_ca_base_enet/wood \
#     --model_ckpt ./results/ad_mar_ca_base_enet/wood/model_latest.pth \
#     --config_path ./results/ad_mar_ca_base_enet/wood/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 1 \
#     --num_samples 1 \
#     --num_inference_steps 50 \
#    --start_step 64 \
#    --sampler ddim \
#    --eta 0.0 \
#    --temperature 10 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_ca_base_enet/zipper \
#     --model_ckpt ./results/ad_mar_ca_base_enet/zipper/model_latest.pth \
#     --config_path ./results/ad_mar_ca_base_enet/zipper/config.yaml \
#     --device cuda \
#     --batch_size 1
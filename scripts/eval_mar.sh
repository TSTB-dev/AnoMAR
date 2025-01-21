# # # bottle  cable  capsule  carpet  grid  hazelnut  leather  metal_nut  pill  screw  tile  toothbrush  transistor  wood  zipper

export CUDA_VISIBLE_DEVICES=4
python src/evaluate_mar.py \
    --num_masks 4 \
    --num_samples 4 \
    --num_inference_steps 100 \
   --start_step 64 \
    --recon_space latent \
    --aggregation mean \
    --output_dir ./results/ad_mar_ca_base_vae_loco_n1/breakfast_box \
    --model_ckpt ./results/ad_mar_ca_base_vae_loco_n1/breakfast_box/model_latest.pth \
    --config_path ./results/ad_mar_ca_base_vae_loco_n1/breakfast_box/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1 \
    --sample_indices 10 20

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
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
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
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
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
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
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
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
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
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
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
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
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
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
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
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
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
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
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#    --start_step 64 \
#    --sampler ddim \
#    --eta 0.0 \
#    --temperature 10 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_ca_base_enet/screw \
#     --model_ckpt ./results/ad_mar_ca_base_enet/screw/model_latest.pth \
#     --config_path ./results/ad_mar_ca_base_enet/screw/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
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
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
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
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
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
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
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
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
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
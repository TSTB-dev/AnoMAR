# # # bottle  cable  capsule  carpet  grid  hazelnut  leather  metal_nut  pill  screw  tile  toothbrush  transistor  wood  zipper

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_emar.py \
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#    --start_step 64 \
#     --recon_space latent \
#     --output_dir ./results/emar_ca_base_enet_loco_enc_id/breakfast_box \
#     --model_ckpt ./results/emar_ca_base_enet_loco_enc_id/breakfast_box/model_latest.pth \
#     --config_path ./results/emar_ca_base_enet_loco_enc_id/breakfast_box/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --sample_indices 0 1 

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_emar.py \
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#    --start_step 64 \
#     --recon_space latent \
#     --output_dir ./results/emar_ca_base_enet_loco_enc_id/juice_bottle \
#     --model_ckpt ./results/emar_ca_base_enet_loco_enc_id/juice_bottle/model_latest.pth \
#     --config_path ./results/emar_ca_base_enet_loco_enc_id/juice_bottle/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --sample_indices 0 1 

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_emar.py \
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#    --start_step 64 \
#     --recon_space latent \
#     --output_dir ./results/emar_ca_base_enet_loco_enc_id/pushpins \
#     --model_ckpt ./results/emar_ca_base_enet_loco_enc_id/pushpins/model_latest.pth \
#     --config_path ./results/emar_ca_base_enet_loco_enc_id/pushpins/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --sample_indices 0 1 

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_emar.py \
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#    --start_step 64 \
#     --recon_space latent \
#     --output_dir ./results/emar_ca_base_enet_loco_enc_id/screw_bag \
#     --model_ckpt ./results/emar_ca_base_enet_loco_enc_id/screw_bag/model_latest.pth \
#     --config_path ./results/emar_ca_base_enet_loco_enc_id/screw_bag/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --sample_indices 0 1 

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_emar.py \
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#    --start_step 64 \
#     --recon_space latent \
#     --output_dir ./results/emar_ca_base_enet_loco_enc_id/splicing_connectors \
#     --model_ckpt ./results/emar_ca_base_enet_loco_enc_id/splicing_connectors/model_latest.pth \
#     --config_path ./results/emar_ca_base_enet_loco_enc_id/splicing_connectors/config.yaml \
#     --device cuda \
#     --batch_size 1 \
#     --sample_indices 0 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_emar.py \
    --num_masks 4 \
    --num_samples 4 \
    --num_inference_steps 100 \
   --start_step 64 \
    --recon_space latent \
    --output_dir ./results/emar_ca_base_vae_ad_enc_id/bottle \
    --model_ckpt ./results/emar_ca_base_vae_ad_enc_id/bottle/model_latest.pth \
    --config_path ./results/emar_ca_base_vae_ad_enc_id/bottle/config.yaml \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_emar.py \
    --num_masks 4 \
    --num_samples 4 \
    --num_inference_steps 100 \
   --start_step 64 \
    --recon_space latent \
    --output_dir ./results/emar_ca_base_vae_ad_enc_id/cable \
    --model_ckpt ./results/emar_ca_base_vae_ad_enc_id/cable/model_latest.pth \
    --config_path ./results/emar_ca_base_vae_ad_enc_id/cable/config.yaml \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_emar.py \
    --num_masks 4 \
    --num_samples 4 \
    --num_inference_steps 100 \
   --start_step 64 \
    --recon_space latent \
    --output_dir ./results/emar_ca_base_vae_ad_enc_id/capsule \
    --model_ckpt ./results/emar_ca_base_vae_ad_enc_id/capsule/model_latest.pth \
    --config_path ./results/emar_ca_base_vae_ad_enc_id/capsule/config.yaml \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_emar.py \
    --num_masks 4 \
    --num_samples 4 \
    --num_inference_steps 100 \
   --start_step 64 \
   --sampler ddim \
   --eta 0.0 \
   --temperature 10 \
    --recon_space latent \
    --output_dir ./results/emar_ca_base_vae_ad_enc_id/carpet \
    --model_ckpt ./results/emar_ca_base_vae_ad_enc_id/carpet/model_latest.pth \
    --config_path ./results/emar_ca_base_vae_ad_enc_id/carpet/config.yaml \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_emar.py \
    --num_masks 4 \
    --num_samples 4 \
    --num_inference_steps 100 \
   --start_step 64 \
    --recon_space latent \
    --output_dir ./results/emar_ca_base_vae_ad_enc_id/grid \
    --model_ckpt ./results/emar_ca_base_vae_ad_enc_id/grid/model_latest.pth \
    --config_path ./results/emar_ca_base_vae_ad_enc_id/grid/config.yaml \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_emar.py \
    --num_masks 4 \
    --num_samples 4 \
    --num_inference_steps 100 \
   --start_step 64 \
    --recon_space latent \
    --output_dir ./results/emar_ca_base_vae_ad_enc_id/hazelnut \
    --model_ckpt ./results/emar_ca_base_vae_ad_enc_id/hazelnut/model_latest.pth \
    --config_path ./results/emar_ca_base_vae_ad_enc_id/hazelnut/config.yaml \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_emar.py \
    --num_masks 4 \
    --num_samples 4 \
    --num_inference_steps 100 \
   --start_step 64 \
    --recon_space latent \
    --output_dir ./results/emar_ca_base_vae_ad_enc_id/leather \
    --model_ckpt ./results/emar_ca_base_vae_ad_enc_id/leather/model_latest.pth \
    --config_path ./results/emar_ca_base_vae_ad_enc_id/leather/config.yaml \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_emar.py \
    --num_masks 4 \
    --num_samples 4 \
    --num_inference_steps 100 \
   --start_step 64 \
    --recon_space latent \
    --output_dir ./results/emar_ca_base_vae_ad_enc_id/metal_nut \
    --model_ckpt ./results/emar_ca_base_vae_ad_enc_id/metal_nut/model_latest.pth \
    --config_path ./results/emar_ca_base_vae_ad_enc_id/metal_nut/config.yaml \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_emar.py \
    --num_masks 4 \
    --num_samples 4 \
    --num_inference_steps 100 \
   --start_step 64 \
    --recon_space latent \
    --output_dir ./results/emar_ca_base_vae_ad_enc_id/pill \
    --model_ckpt ./results/emar_ca_base_vae_ad_enc_id/pill/model_latest.pth \
    --config_path ./results/emar_ca_base_vae_ad_enc_id/pill/config.yaml \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_emar.py \
    --num_masks 4 \
    --num_samples 4 \
    --num_inference_steps 100 \
   --start_step 64 \
    --recon_space latent \
    --output_dir ./results/emar_ca_base_vae_ad_enc_id/screw \
    --model_ckpt ./results/emar_ca_base_vae_ad_enc_id/screw/model_latest.pth \
    --config_path ./results/emar_ca_base_vae_ad_enc_id/screw/config.yaml \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_emar.py \
    --num_masks 4 \
    --num_samples 4 \
    --num_inference_steps 100 \
   --start_step 64 \
    --recon_space latent \
    --output_dir ./results/emar_ca_base_vae_ad_enc_id/tile \
    --model_ckpt ./results/emar_ca_base_vae_ad_enc_id/tile/model_latest.pth \
    --config_path ./results/emar_ca_base_vae_ad_enc_id/tile/config.yaml \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_emar.py \
    --num_masks 4 \
    --num_samples 4 \
    --num_inference_steps 100 \
   --start_step 64 \
    --recon_space latent \
    --output_dir ./results/emar_ca_base_vae_ad_enc_id/toothbrush \
    --model_ckpt ./results/emar_ca_base_vae_ad_enc_id/toothbrush/model_latest.pth \
    --config_path ./results/emar_ca_base_vae_ad_enc_id/toothbrush/config.yaml \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_emar.py \
    --num_masks 4 \
    --num_samples 4 \
    --num_inference_steps 100 \
   --start_step 64 \
    --recon_space latent \
    --output_dir ./results/emar_ca_base_vae_ad_enc_id/transistor \
    --model_ckpt ./results/emar_ca_base_vae_ad_enc_id/transistor/model_latest.pth \
    --config_path ./results/emar_ca_base_vae_ad_enc_id/transistor/config.yaml

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_emar.py \
    --num_masks 4 \
    --num_samples 4 \
    --num_inference_steps 100 \
   --start_step 64 \
    --recon_space latent \
    --output_dir ./results/emar_ca_base_vae_ad_enc_id/wood \
    --model_ckpt ./results/emar_ca_base_vae_ad_enc_id/wood/model_latest.pth \
    --config_path ./results/emar_ca_base_vae_ad_enc_id/wood/config.yaml \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_emar.py \
    --num_masks 4 \
    --num_samples 4 \
    --num_inference_steps 100 \
   --start_step 64 \
    --recon_space latent \
    --output_dir ./results/emar_ca_base_vae_ad_enc_id/zipper \
    --model_ckpt ./results/emar_ca_base_vae_ad_enc_id/zipper/model_latest.pth \
    --config_path ./results/emar_ca_base_vae_ad_enc_id/zipper/config.yaml \
    --device cuda \
    --batch_size 1
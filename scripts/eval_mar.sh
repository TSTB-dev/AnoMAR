# # # bottle  cable  capsule  carpet  grid  hazelnut  leather  metal_nut  pill  screw  tile  toothbrush  transistor  wood  zipper

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_mar.py \
    --num_masks 4 \
    --num_samples 1 \
    --num_inference_steps 100 \
    --start_step 64 \
    --recon_space latent \
    --output_dir ./results/ad_mar_ca_base/cable \
    --model_ckpt ./results/ad_mar_ca_base/cable/model_latest.pth \
    --config_path ./results/ad_mar_ca_base/cable/config.yaml \
    --device cuda \
    --save_images \
    --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --start_step 16 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_adaln_base/cable \
#     --model_ckpt ./results/ad_mar_adaln_base/cable/model_latest.pth \
#     --config_path ./results/ad_mar_adaln_base/cable/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --start_step 16 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_adaln_base/capsule \
#     --model_ckpt ./results/ad_mar_adaln_base/capsule/model_latest.pth \
#     --config_path ./results/ad_mar_adaln_base/capsule/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --start_step 16 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_adaln_base/carpet \
#     --model_ckpt ./results/ad_mar_adaln_base/carpet/model_latest.pth \
#     --config_path ./results/ad_mar_adaln_base/carpet/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --start_step 16 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_adaln_base/grid \
#     --model_ckpt ./results/ad_mar_adaln_base/grid/model_latest.pth \
#     --config_path ./results/ad_mar_adaln_base/grid/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --start_step 16 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_adaln_base/hazelnut \
#     --model_ckpt ./results/ad_mar_adaln_base/hazelnut/model_latest.pth \
#     --config_path ./results/ad_mar_adaln_base/hazelnut/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --start_step 16 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_adaln_base/leather \
#     --model_ckpt ./results/ad_mar_adaln_base/leather/model_latest.pth \
#     --config_path ./results/ad_mar_adaln_base/leather/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --start_step 16 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_adaln_base/metal_nut \
#     --model_ckpt ./results/ad_mar_adaln_base/metal_nut/model_ema_latest.pth \
#     --config_path ./results/ad_mar_adaln_base/metal_nut/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --start_step 16 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_adaln_base/pill \
#     --model_ckpt ./results/ad_mar_adaln_base/pill/model_latest.pth \
#     --config_path ./results/ad_mar_adaln_base/pill/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --start_step 16 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_adaln_base/screw \
#     --model_ckpt ./results/ad_mar_adaln_base/screw/model_latest.pth \
#     --config_path ./results/ad_mar_adaln_base/screw/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --start_step 16 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_adaln_base/tile \
#     --model_ckpt ./results/ad_mar_adaln_base/tile/model_latest.pth \
#     --config_path ./results/ad_mar_adaln_base/tile/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --start_step 16 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_adaln_base/toothbrush \
#     --model_ckpt ./results/ad_mar_adaln_base/toothbrush/model_latest.pth \
#     --config_path ./results/ad_mar_adaln_base/toothbrush/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --start_step 16 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_adaln_base/transistor \
#     --model_ckpt ./results/ad_mar_adaln_base/transistor/model_latest.pth \
#     --config_path ./results/ad_mar_adaln_base/transistor/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --start_step 16 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_adaln_base/wood \
#     --model_ckpt ./results/ad_mar_adaln_base/wood/model_latest.pth \
#     --config_path ./results/ad_mar_adaln_base/wood/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mar.py \
#     --num_masks 4 \
#     --num_samples 4 \
#     --num_inference_steps 100 \
#     --start_step 16 \
#     --recon_space latent \
#     --output_dir ./results/ad_mar_adaln_base/zipper \
#     --model_ckpt ./results/ad_mar_adaln_base/zipper/model_latest.pth \
#     --config_path ./results/ad_mar_adaln_base/zipper/config.yaml \
#     --device cuda \
#     --save_images \
#     --batch_size 1

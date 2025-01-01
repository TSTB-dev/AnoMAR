# # bottle  cable  capsule  carpet  grid  hazelnut  leather  metal_nut  pill  screw  tile  toothbrush  transistor  wood  zipper

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_mim.py \
   --num_masks 4 \
    --recon_space latent \
    --save_images \
    --output_dir ./results/ad_mim_base_long/bottle \
    --model_ckpt ./results/ad_mim_base_long/bottle/model_latest.pth \
    --config_path ./results/ad_mim_base_long/bottle/config.yaml \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_mim.py \
   --num_masks 4 \
    --recon_space latent \
    --save_images \
    --output_dir ./results/ad_mim_base_long/cable \
    --model_ckpt ./results/ad_mim_base_long/cable/model_latest.pth \
    --config_path ./results/ad_mim_base_long/cable/config.yaml \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_mim.py \
   --num_masks 4 \
    --recon_space latent \
    --save_images \
    --output_dir ./results/ad_mim_base_long/capsule \
    --model_ckpt ./results/ad_mim_base_long/capsule/model_latest.pth \
    --config_path ./results/ad_mim_base_long/capsule/config.yaml \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_mim.py \
   --num_masks 4 \
    --recon_space latent \
    --save_images \
    --output_dir ./results/ad_mim_base_long/carpet \
    --model_ckpt ./results/ad_mim_base_long/carpet/model_latest.pth \
    --config_path ./results/ad_mim_base_long/carpet/config.yaml \
    --device cuda \
    --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 32 \
#     --recon_space latent \
#     --save_images \
#     --output_dir ./results/ad_mim_base/bottle \
#     --model_ckpt ./results/ad_mim_base/bottle/model_latest.pth \
#     --config_path ./results/ad_mim_base/bottle/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 32 \
#     --recon_space latent \
#     --save_images \
#     --output_dir ./results/ad_mim_base/cable \
#     --model_ckpt ./results/ad_mim_base/cable/model_latest.pth \
#     --config_path ./results/ad_mim_base/cable/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 32 \
#     --recon_space latent \
#     --save_images \
#     --output_dir ./results/ad_mim_base/capsule \
#     --model_ckpt ./results/ad_mim_base/capsule/model_latest.pth \
#     --config_path ./results/ad_mim_base/capsule/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 32 \
#     --recon_space latent \
#     --save_images \
#     --output_dir ./results/ad_mim_base/carpet \
#     --model_ckpt ./results/ad_mim_base/carpet/model_latest.pth \
#     --config_path ./results/ad_mim_base/carpet/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 32 \
#     --recon_space latent \
#     --save_images \
#     --output_dir ./results/ad_mim_base/grid \
#     --model_ckpt ./results/ad_mim_base/grid/model_latest.pth \
#     --config_path ./results/ad_mim_base/grid/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 32 \
#     --recon_space latent \
#     --save_images \
#     --output_dir ./results/ad_mim_base/hazelnut \
#     --model_ckpt ./results/ad_mim_base/hazelnut/model_latest.pth \
#     --config_path ./results/ad_mim_base/hazelnut/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 32 \
#     --recon_space latent \
#     --save_images \
#     --output_dir ./results/ad_mim_base/leather \
#     --model_ckpt ./results/ad_mim_base/leather/model_latest.pth \
#     --config_path ./results/ad_mim_base/leather/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 32 \
#     --recon_space latent \
#     --save_images \
#     --output_dir ./results/ad_mim_base/metal_nut \
#     --model_ckpt ./results/ad_mim_base/metal_nut/model_latest.pth \
#     --config_path ./results/ad_mim_base/metal_nut/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 32 \
#     --recon_space latent \
#     --save_images \
#     --output_dir ./results/ad_mim_base/pill \
#     --model_ckpt ./results/ad_mim_base/pill/model_latest.pth \
#     --config_path ./results/ad_mim_base/pill/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 32 \
#     --recon_space latent \
#     --save_images \
#     --output_dir ./results/ad_mim_base/screw \
#     --model_ckpt ./results/ad_mim_base/screw/model_latest.pth \
#     --config_path ./results/ad_mim_base/screw/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 32 \
#     --recon_space latent \
#     --save_images \
#     --output_dir ./results/ad_mim_base/tile \
#     --model_ckpt ./results/ad_mim_base/tile/model_latest.pth \
#     --config_path ./results/ad_mim_base/tile/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 32 \
#     --recon_space latent \
#     --save_images \
#     --output_dir ./results/ad_mim_base/toothbrush \
#     --model_ckpt ./results/ad_mim_base/toothbrush/model_latest.pth \
#     --config_path ./results/ad_mim_base/toothbrush/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 32 \
#     --recon_space latent \
#     --save_images \
#     --output_dir ./results/ad_mim_base/transistor \
#     --model_ckpt ./results/ad_mim_base/transistor/model_latest.pth \
#     --config_path ./results/ad_mim_base/transistor/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 32 \
#     --recon_space latent \
#     --save_images \
#     --output_dir ./results/ad_mim_base/wood \
#     --model_ckpt ./results/ad_mim_base/wood/model_latest.pth \
#     --config_path ./results/ad_mim_base/wood/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 32 \
#     --recon_space latent \
#     --save_images \
#     --output_dir ./results/ad_mim_base/zipper \
#     --model_ckpt ./results/ad_mim_base/zipper/model_latest.pth \
#     --config_path ./results/ad_mim_base/zipper/config.yaml \
#     --device cuda \
#     --batch_size 1

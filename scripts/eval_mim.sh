# # # bottle  cable  capsule  carpet  grid  hazelnut  leather  metal_nut  pill  screw  tile  toothbrush  transistor  wood  zipper

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_mim.py \
   --num_masks 4 \
    --recon_space latent \
    --output_dir ./results/loco_mim_base_vae/breakfast_box \
    --model_ckpt ./results/loco_mim_base_vae/breakfast_box/model_latest.pth \
    --config_path ./results/loco_mim_base_vae/breakfast_box/config.yaml \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_mim.py \
   --num_masks 4 \
    --recon_space latent \
    --output_dir ./results/loco_mim_base_vae/juice_bottle \
    --model_ckpt ./results/loco_mim_base_vae/juice_bottle/model_latest.pth \
    --config_path ./results/loco_mim_base_vae/juice_bottle/config.yaml \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_mim.py \
   --num_masks 4 \
    --recon_space latent \
    --output_dir ./results/loco_mim_base_vae/pushpins \
    --model_ckpt ./results/loco_mim_base_vae/pushpins/model_latest.pth \
    --config_path ./results/loco_mim_base_vae/pushpins/config.yaml \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_mim.py \
   --num_masks 4 \
    --recon_space latent \
    --output_dir ./results/loco_mim_base_vae/screw_bag \
    --model_ckpt ./results/loco_mim_base_vae/screw_bag/model_latest.pth \
    --config_path ./results/loco_mim_base_vae/screw_bag/config.yaml \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_mim.py \
   --num_masks 4 \
    --recon_space latent \
    --output_dir ./results/loco_mim_base_vae/splicing_connectors \
    --model_ckpt ./results/loco_mim_base_vae/splicing_connectors/model_latest.pth \
    --config_path ./results/loco_mim_base_vae/splicing_connectors/config.yaml \
    --device cuda \
    --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 4 \
#     --recon_space latent \
#     --output_dir ./results/ad_mim_base_enet/bottle \
#     --model_ckpt ./results/ad_mim_base_enet/bottle/model_ema_latest.pth \
#     --config_path ./results/ad_mim_base_enet/bottle/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 4 \
#     --recon_space latent \
#     --output_dir ./results/ad_mim_base_enet/cable \
#     --model_ckpt ./results/ad_mim_base_enet/cable/model_ema_latest.pth \
#     --config_path ./results/ad_mim_base_enet/cable/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 4 \
#     --recon_space latent \
#     --output_dir ./results/ad_mim_base_enet/capsule \
#     --model_ckpt ./results/ad_mim_base_enet/capsule/model_ema_latest.pth \
#     --config_path ./results/ad_mim_base_enet/capsule/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 4 \
#     --recon_space latent \
#     --output_dir ./results/ad_mim_base_enet/carpet \
#     --model_ckpt ./results/ad_mim_base_enet/carpet/model_ema_latest.pth \
#     --config_path ./results/ad_mim_base_enet/carpet/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 4 \
#     --recon_space latent \
#     --output_dir ./results/ad_mim_base_enet/grid \
#     --model_ckpt ./results/ad_mim_base_enet/grid/model_ema_latest.pth \
#     --config_path ./results/ad_mim_base_enet/grid/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 4 \
#     --recon_space latent \
#     --output_dir ./results/ad_mim_base_enet/hazelnut \
#     --model_ckpt ./results/ad_mim_base_enet/hazelnut/model_ema_latest.pth \
#     --config_path ./results/ad_mim_base_enet/hazelnut/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 4 \
#     --recon_space latent \
#     --output_dir ./results/ad_mim_base_enet/leather \
#     --model_ckpt ./results/ad_mim_base_enet/leather/model_ema_latest.pth \
#     --config_path ./results/ad_mim_base_enet/leather/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 4 \
#     --recon_space latent \
#     --output_dir ./results/ad_mim_base_enet/metal_nut \
#     --model_ckpt ./results/ad_mim_base_enet/metal_nut/model_ema_latest.pth \
#     --config_path ./results/ad_mim_base_enet/metal_nut/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 4 \
#     --recon_space latent \
#     --output_dir ./results/ad_mim_base_enet/pill \
#     --model_ckpt ./results/ad_mim_base_enet/pill/model_ema_latest.pth \
#     --config_path ./results/ad_mim_base_enet/pill/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 4 \
#     --recon_space latent \
#     --output_dir ./results/ad_mim_base_enet/screw \
#     --model_ckpt ./results/ad_mim_base_enet/screw/model_ema_latest.pth \
#     --config_path ./results/ad_mim_base_enet/screw/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 4 \
#     --recon_space latent \
#     --output_dir ./results/ad_mim_base_enet/tile \
#     --model_ckpt ./results/ad_mim_base_enet/tile/model_ema_latest.pth \
#     --config_path ./results/ad_mim_base_enet/tile/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 4 \
#     --recon_space latent \
#     --output_dir ./results/ad_mim_base_enet/toothbrush \
#     --model_ckpt ./results/ad_mim_base_enet/toothbrush/model_ema_latest.pth \
#     --config_path ./results/ad_mim_base_enet/toothbrush/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 4 \
#     --recon_space latent \
#     --output_dir ./results/ad_mim_base_enet/transistor \
#     --model_ckpt ./results/ad_mim_base_enet/transistor/model_ema_latest.pth \
#     --config_path ./results/ad_mim_base_enet/transistor/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 4 \
#     --recon_space latent \
#     --output_dir ./results/ad_mim_base_enet/wood \
#     --model_ckpt ./results/ad_mim_base_enet/wood/model_ema_latest.pth \
#     --config_path ./results/ad_mim_base_enet/wood/config.yaml \
#     --device cuda \
#     --batch_size 1

# export CUDA_VISIBLE_DEVICES=0
# python src/evaluate_mim.py \
#    --num_masks 4 \
#     --recon_space latent \
#     --output_dir ./results/ad_mim_base_enet/zipper \
#     --model_ckpt ./results/ad_mim_base_enet/zipper/model_ema_latest.pth \
#     --config_path ./results/ad_mim_base_enet/zipper/config.yaml \
#     --device cuda \
#     --batch_size 1

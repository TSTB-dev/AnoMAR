# CUDA_VISIBLE_DEVICES=4
# python3 ./src/train_dit.py --config_path ./configs/exp_dit_vae_ad/bottle.yml

# CUDA_VISIBLE_DEVICES=4 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_smc/bottle.yml 

# CUDA_VISIBLE_DEVICES=5 python3 ./src/train_dit.py --config_path ./configs/exp_dit_vae_ad/capsule.yml & \
# CUDA_VISIBLE_DEVICES=6 python3 ./src/train_dit.py --config_path ./configs/exp_dit_vae_ad/bottle.yml & \
# CUDA_VISIBLE_DEVICES=7 python3 ./src/train_dit.py --config_path ./configs/exp_dit_vae_ad/carpet.yml 

# CUDA_VISIBLE_DEVICES=5 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_vae_ad_norm/grid.yml
# CUDA_VISIBLE_DEVICES=6 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_vae_ad/bottle.yml & \
# CUDA_VISIBLE_DEVICES=7 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_vae_ad/carpet.yml & \

# candle.yml  capsules.yml  cashew.yml  chewinggum.yml  fryum.yml  macaroni1.yml  macaroni2.yml  pcb1.yml  pcb2.yml  pcb3.yml  pcb4.yml  pipe_fryum.yml

CUDA_VISIBLE_DEVICES=0 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_es/bottle.yml & \
CUDA_VISIBLE_DEVICES=0 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_es/cable.yml & \

CUDA_VISIBLE_DEVICES=1 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_es/capsule.yml & \
CUDA_VISIBLE_DEVICES=1 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_es/carpet.yml & \

CUDA_VISIBLE_DEVICES=2 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_es/grid.yml & \
CUDA_VISIBLE_DEVICES=2 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_es/hazelnut.yml & \

CUDA_VISIBLE_DEVICES=3 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_es/metal_nut.yml & \
CUDA_VISIBLE_DEVICES=3 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_es/pill.yml & \

CUDA_VISIBLE_DEVICES=4 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_es/screw.yml & \
CUDA_VISIBLE_DEVICES=4 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_es/tile.yml & \

CUDA_VISIBLE_DEVICES=5 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_es/transistor.yml & \
CUDA_VISIBLE_DEVICES=5 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_es/wood.yml & \

CUDA_VISIBLE_DEVICES=6 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_es/toothbrush.yml & \
CUDA_VISIBLE_DEVICES=6 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_es/zipper.yml & \

CUDA_VISIBLE_DEVICES=7 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_es/leather.yml & \
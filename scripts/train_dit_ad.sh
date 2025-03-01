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

CUDA_VISIBLE_DEVICES=0 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad/bottle.yml & \
CUDA_VISIBLE_DEVICES=0 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad/cable.yml & \

CUDA_VISIBLE_DEVICES=1 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad/capsule.yml & \
CUDA_VISIBLE_DEVICES=1 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad/carpet.yml & \

CUDA_VISIBLE_DEVICES=2 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad/grid.yml & \
CUDA_VISIBLE_DEVICES=2 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad/hazelnut.yml & \

CUDA_VISIBLE_DEVICES=3 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad/metal_nut.yml & \
CUDA_VISIBLE_DEVICES=3 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad/pill.yml & \

CUDA_VISIBLE_DEVICES=4 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad/screw.yml & \
CUDA_VISIBLE_DEVICES=4 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad/tile.yml & \

CUDA_VISIBLE_DEVICES=5 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad/transistor.yml & \
CUDA_VISIBLE_DEVICES=5 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad/wood.yml & \

CUDA_VISIBLE_DEVICES=6 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad/toothbrush.yml & \
CUDA_VISIBLE_DEVICES=6 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad/zipper.yml & \

CUDA_VISIBLE_DEVICES=7 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad/leather.yml 

# CUDA_VISIBLE_DEVICES=6 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_visa/wood.yml & \
# CUDA_VISIBLE_DEVICES=6 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_visa/zipper.yml & \

# CUDA_VISIBLE_DEVICES=7 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_visa/tile.yml 
# /home/haselab/projects/sakai/AnoMAR/AnoMAR/configs/exp_unet_feature_ad
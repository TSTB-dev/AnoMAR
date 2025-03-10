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

# CUDA_VISIBLE_DEVICES=0 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_pdn//bottle.yml & \
# CUDA_VISIBLE_DEVICES=0 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_pdn/cable.yml & \

# CUDA_VISIBLE_DEVICES=1 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_pdn/capsule.yml & \
# CUDA_VISIBLE_DEVICES=1 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_pdn/carpet.yml & \

# CUDA_VISIBLE_DEVICES=2 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_pdn/grid.yml & \
# CUDA_VISIBLE_DEVICES=2 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_pdn/hazelnut.yml & \

# CUDA_VISIBLE_DEVICES=3 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_pdn/metal_nut.yml & \
# CUDA_VISIBLE_DEVICES=3 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_pdn/pill.yml & \

# CUDA_VISIBLE_DEVICES=4 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_pdn/screw.yml & \
# CUDA_VISIBLE_DEVICES=4 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_pdn/tile.yml & \

# CUDA_VISIBLE_DEVICES=5 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_pdn/transistor.yml & \
# CUDA_VISIBLE_DEVICES=5 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_pdn/wood.yml & \

# CUDA_VISIBLE_DEVICES=6 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_pdn/toothbrush.yml & \
# CUDA_VISIBLE_DEVICES=6 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_pdn/zipper.yml & \

# CUDA_VISIBLE_DEVICES=7 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_pdn/leather.yml & \

CUDA_VISIBLE_DEVICES=0 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_visa_es/candle.yml & \
CUDA_VISIBLE_DEVICES=0 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_visa_es/capsules.yml & \

CUDA_VISIBLE_DEVICES=1 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_visa_es/cashew.yml & \
CUDA_VISIBLE_DEVICES=1 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_visa_es/chewinggum.yml & \

CUDA_VISIBLE_DEVICES=2 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_visa_es/fryum.yml & \
CUDA_VISIBLE_DEVICES=2 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_visa_es/macaroni1.yml & \

CUDA_VISIBLE_DEVICES=3 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_visa_es/macaroni2.yml & \
CUDA_VISIBLE_DEVICES=3 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_visa_es/pcb1.yml & \

CUDA_VISIBLE_DEVICES=4 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_visa_es/pcb2.yml & \
CUDA_VISIBLE_DEVICES=4 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_visa_es/pcb3.yml & \

CUDA_VISIBLE_DEVICES=5 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_visa_es/pcb4.yml & \
CUDA_VISIBLE_DEVICES=5 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_visa_es/pipe_fryum.yml & \

CUDA_VISIBLE_DEVICES=6 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_es/capsule.yml & \
CUDA_VISIBLE_DEVICES=6 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_es/metal_nut.yml & \

CUDA_VISIBLE_DEVICES=7 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_es/pill.yml & \
CUDA_VISIBLE_DEVICES=7 python3 ./src/train_dit_feature.py --config_path ./configs/exp_unet_feature_ad_es/screw.yml & \


# CUDA_VISIBLE_DEVICES=4
# python3 ./src/train_dit.py --config_path ./configs/exp_dit_vae_ad/bottle.yml

# CUDA_VISIBLE_DEVICES=4 python3 ./src/train_dit.py --config_path ./configs/exp_dit_vae_ad/screw.yml & \
# CUDA_VISIBLE_DEVICES=5 python3 ./src/train_dit.py --config_path ./configs/exp_dit_vae_ad/capsule.yml & \
# CUDA_VISIBLE_DEVICES=6 python3 ./src/train_dit.py --config_path ./configs/exp_dit_vae_ad/bottle.yml & \
# CUDA_VISIBLE_DEVICES=7 python3 ./src/train_dit.py --config_path ./configs/exp_dit_vae_ad/carpet.yml 

CUDA_VISIBLE_DEVICES=0 python3 ./src/train_dit.py --config_path ./configs/exp_unet_vae_ad/grid.yml 
CUDA_VISIBLE_DEVICES=1 python3 ./src/train_dit.py --config_path ./configs/exp_unet_vae_ad/screw.yml
CUDA_VISIBLE_DEVICES=2 python3 ./src/train_dit.py --config_path ./configs/exp_unet_vae_ad/bottle.yml
CUDA_VISIBLE_DEVICES=3 python3 ./src/train_dit.py --config_path ./configs/exp_unet_vae_ad/carpet.yml
CUDA_VISIBLE_DEVICES=4 python3 ./src/train_dit.py --config_path ./configs/exp_dit_vae_ad/grid.yml
CUDA_VISIBLE_DEVICES=5 python3 ./src/train_dit.py --config_path ./configs/exp_dit_vae_ad/screw.yml
CUDA_VISIBLE_DEVICES=6 python3 ./src/train_dit.py --config_path ./configs/exp_dit_vae_ad/bottle.yml
CUDA_VISIBLE_DEVICES=7 python3 ./src/train_dit.py --config_path ./configs/exp_dit_vae_ad/carpet.yml
export CUDA_VISIBLE_DEVICES=0
python src/train_da.py \
    --num_epochs 20 \
    --batch_size 16 \
    --lr 0.01 \
    --diffusion_config ./config_bottle.yaml \
    --diffusion_ckpt ./ddad_models/leather \
    --backbone_name enet \
    --mask_strategy checkerboard \
    --num_inference_steps 100 \
    --num_samples 1 \
    --save_dir ./results_da/bottle & \

export CUDA_VISIBLE_DEVICES=1
python src/train_da.py \
    --num_epochs 20 \
    --batch_size 16 \
    --lr 0.01 \
    --diffusion_config ./config_cable.yaml \
    --diffusion_ckpt ./ddad_models/cable \
    --backbone_name enet \
    --mask_strategy checkerboard \
    --num_inference_steps 100 \
    --num_samples 1 \
    --save_dir ./results_da/cable & \

export CUDA_VISIBLE_DEVICES=2
python src/train_da.py \
    --num_epochs 20 \
    --batch_size 16 \
    --lr 0.01 \
    --diffusion_config ./config_capsule.yaml \
    --diffusion_ckpt ./ddad_models/capsule \
    --backbone_name enet \
    --mask_strategy checkerboard \
    --num_inference_steps 100 \
    --num_samples 1 \
    --save_dir ./results_da/capsule & \

export CUDA_VISIBLE_DEVICES=3
python src/train_da.py \
    --num_epochs 20 \
    --batch_size 16 \
    --lr 0.01 \
    --diffusion_config ./config_carpet.yaml \
    --diffusion_ckpt ./ddad_models/carpet \
    --backbone_name enet \
    --mask_strategy checkerboard \
    --num_inference_steps 100 \
    --num_samples 1 \
    --save_dir ./results_da/carpet & \

export CUDA_VISIBLE_DEVICES=4
python src/train_da.py \
    --num_epochs 20 \
    --batch_size 16 \
    --lr 0.01 \
    --diffusion_config ./config_grid.yaml \
    --diffusion_ckpt ./ddad_models/grid \
    --backbone_name enet \
    --mask_strategy checkerboard \
    --num_inference_steps 100 \
    --num_samples 1 \
    --save_dir ./results_da/grid & \

export CUDA_VISIBLE_DEVICES=5
python src/train_da.py \
    --num_epochs 20 \
    --batch_size 16 \
    --lr 0.01 \
    --diffusion_config ./config_hazelnut.yaml \
    --diffusion_ckpt ./ddad_models/hazelnut \
    --backbone_name enet \
    --mask_strategy checkerboard \
    --num_inference_steps 100 \
    --num_samples 1 \
    --save_dir ./results_da/hazelnut & \

export CUDA_VISIBLE_DEVICES=6
python src/train_da.py \
    --num_epochs 20 \
    --batch_size 16 \
    --lr 0.01 \
    --diffusion_config ./config_leather.yaml \
    --diffusion_ckpt ./ddad_models/leather \
    --backbone_name enet \
    --mask_strategy checkerboard \
    --num_inference_steps 100 \
    --num_samples 1 \
    --save_dir ./results_da/leather & \

export CUDA_VISIBLE_DEVICES=7
python src/train_da.py \
    --num_epochs 20 \
    --batch_size 16 \
    --lr 0.01 \
    --diffusion_config ./config_metal_nut.yaml \
    --diffusion_ckpt ./ddad_models/metal_nut \
    --backbone_name enet \
    --mask_strategy checkerboard \
    --num_inference_steps 100 \
    --num_samples 1 \
    --save_dir ./results_da/metal_nut & \

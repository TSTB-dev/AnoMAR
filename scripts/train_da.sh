export CUDA_VISIBLE_DEVICES=0
python src/train_da.py \
    --num_epochs 20 \
    --batch_size 16 \
    --lr 0.01 \
    --diffusion_config ./config_bottle.yaml \
    --diffusion_ckpt ./ddad_models/bottle \
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
    --diffusion_config ./config_zipper.yaml \
    --diffusion_ckpt ./ddad_models/zipper \
    --backbone_name enet \
    --mask_strategy checkerboard \
    --num_inference_steps 100 \
    --num_samples 1 \
    --save_dir ./results_da/zipper & \

export CUDA_VISIBLE_DEVICES=2
python src/train_da.py \
    --num_epochs 20 \
    --batch_size 16 \
    --lr 0.01 \
    --diffusion_config ./config_wood.yaml \
    --diffusion_ckpt ./ddad_models/wood \
    --backbone_name enet \
    --mask_strategy checkerboard \
    --num_inference_steps 100 \
    --num_samples 1 \
    --save_dir ./results_da/wood & \

export CUDA_VISIBLE_DEVICES=3
python src/train_da.py \
    --num_epochs 20 \
    --batch_size 16 \
    --lr 0.01 \
    --diffusion_config ./config_transistor.yaml \
    --diffusion_ckpt ./ddad_models/transistor \
    --backbone_name enet \
    --mask_strategy checkerboard \
    --num_inference_steps 100 \
    --num_samples 1 \
    --save_dir ./results_da/transistor & \

export CUDA_VISIBLE_DEVICES=4
python src/train_da.py \
    --num_epochs 20 \
    --batch_size 16 \
    --lr 0.01 \
    --diffusion_config ./config_toothbrush.yaml \
    --diffusion_ckpt ./ddad_models/toothbrush \
    --backbone_name enet \
    --mask_strategy checkerboard \
    --num_inference_steps 100 \
    --num_samples 1 \
    --save_dir ./results_da/toothbrush & \

export CUDA_VISIBLE_DEVICES=5
python src/train_da.py \
    --num_epochs 20 \
    --batch_size 16 \
    --lr 0.01 \
    --diffusion_config ./config_tile.yaml \
    --diffusion_ckpt ./ddad_models/tile \
    --backbone_name enet \
    --mask_strategy checkerboard \
    --num_inference_steps 100 \
    --num_samples 1 \
    --save_dir ./results_da/tile & \

export CUDA_VISIBLE_DEVICES=6
python src/train_da.py \
    --num_epochs 20 \
    --batch_size 16 \
    --lr 0.01 \
    --diffusion_config ./config_screw.yaml \
    --diffusion_ckpt ./ddad_models/screw \
    --backbone_name enet \
    --mask_strategy checkerboard \
    --num_inference_steps 100 \
    --num_samples 1 \
    --save_dir ./results_da/screw & \

export CUDA_VISIBLE_DEVICES=7
python src/train_da.py \
    --num_epochs 20 \
    --batch_size 16 \
    --lr 0.01 \
    --diffusion_config ./config_pill.yaml \
    --diffusion_ckpt ./ddad_models/pill \
    --backbone_name enet \
    --mask_strategy checkerboard \
    --num_inference_steps 100 \
    --num_samples 1 \
    --save_dir ./results_da/pill & \

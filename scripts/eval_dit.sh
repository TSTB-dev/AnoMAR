# # bottle  cable  capsule  carpet  grid  hazelnut  leather  metal_nut  pill  screw  tile  toothbrush  transistor  wood  zipper

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit.py \
    --num_samples 32 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 8 \
    --output_dir ./results/ad_dit_d8w768/bottle \
    --model_ckpt ./results/ad_dit_d8w768/bottle/model_latest.pth \
    --config_path ./results/ad_dit_d8w768/bottle/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit.py \
    --num_samples 32 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 8 \
    --output_dir ./results/ad_dit_d8w768/cable \
    --model_ckpt ./results/ad_dit_d8w768/cable/model_latest.pth \
    --config_path ./results/ad_dit_d8w768/cable/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit.py \
    --num_samples 32 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 8 \
    --output_dir ./results/ad_dit_d8w768/capsule \
    --model_ckpt ./results/ad_dit_d8w768/capsule/model_latest.pth \
    --config_path ./results/ad_dit_d8w768/capsule/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit.py \
    --num_samples 32 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 8 \
    --output_dir ./results/ad_dit_d8w768/carpet \
    --model_ckpt ./results/ad_dit_d8w768/carpet/model_latest.pth \
    --config_path ./results/ad_dit_d8w768/carpet/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit.py \
    --num_samples 32 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 8 \
    --output_dir ./results/ad_dit_d8w768/grid \
    --model_ckpt ./results/ad_dit_d8w768/grid/model_latest.pth \
    --config_path ./results/ad_dit_d8w768/grid/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit.py \
    --num_samples 32 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 8 \
    --output_dir ./results/ad_dit_d8w768/hazelnut \
    --model_ckpt ./results/ad_dit_d8w768/hazelnut/model_latest.pth \
    --config_path ./results/ad_dit_d8w768/hazelnut/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit.py \
    --num_samples 32 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 8 \
    --output_dir ./results/ad_dit_d8w768/leather \
    --model_ckpt ./results/ad_dit_d8w768/leather/model_latest.pth \
    --config_path ./results/ad_dit_d8w768/leather/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit.py \
    --num_samples 32 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 8 \
    --output_dir ./results/ad_dit_d8w768/metal_nut \
    --model_ckpt ./results/ad_dit_d8w768/metal_nut/model_latest.pth \
    --config_path ./results/ad_dit_d8w768/metal_nut/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit.py \
    --num_samples 32 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 8 \
    --output_dir ./results/ad_dit_d8w768/pill \
    --model_ckpt ./results/ad_dit_d8w768/pill/model_latest.pth \
    --config_path ./results/ad_dit_d8w768/pill/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit.py \
    --num_samples 32 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 8 \
    --output_dir ./results/ad_dit_d8w768/screw \
    --model_ckpt ./results/ad_dit_d8w768/screw/model_latest.pth \
    --config_path ./results/ad_dit_d8w768/screw/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit.py \
    --num_samples 32 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 8 \
    --output_dir ./results/ad_dit_d8w768/tile \
    --model_ckpt ./results/ad_dit_d8w768/tile/model_latest.pth \
    --config_path ./results/ad_dit_d8w768/tile/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit.py \
    --num_samples 32 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 8 \
    --output_dir ./results/ad_dit_d8w768/toothbrush \
    --model_ckpt ./results/ad_dit_d8w768/toothbrush/model_latest.pth \
    --config_path ./results/ad_dit_d8w768/toothbrush/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit.py \
    --num_samples 32 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 8 \
    --output_dir ./results/ad_dit_d8w768/transistor \
    --model_ckpt ./results/ad_dit_d8w768/transistor/model_latest.pth \
    --config_path ./results/ad_dit_d8w768/transistor/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit.py \
    --num_samples 32 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 8 \
    --output_dir ./results/ad_dit_d8w768/wood \
    --model_ckpt ./results/ad_dit_d8w768/wood/model_latest.pth \
    --config_path ./results/ad_dit_d8w768/wood/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit.py \
    --num_samples 32 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 8 \
    --output_dir ./results/ad_dit_d8w768/zipper \
    --model_ckpt ./results/ad_dit_d8w768/zipper/model_latest.pth \
    --config_path ./results/ad_dit_d8w768/zipper/config.yaml \
    --save_images \
    --device cuda \
    --batch_size 1

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit_feature.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 32 \
    --output_dir ./results/exp_unet_feature_visa/candle \
    --model_ckpt ./results/exp_unet_feature_visa/candle/model_ema_latest.pth \
    --config_path ./results/exp_unet_feature_visa/candle/config.yaml \
    --device cuda \
    --batch_size 1 \
    --save_all_images \

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit_feature.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 32 \
    --output_dir ./results/exp_unet_feature_visa/capsules \
    --model_ckpt ./results/exp_unet_feature_visa/capsules/model_ema_latest.pth \
    --config_path ./results/exp_unet_feature_visa/capsules/config.yaml \
    --device cuda \
    --batch_size 1 \
    --save_all_images \

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit_feature.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 32 \
    --output_dir ./results/exp_unet_feature_visa/cashew \
    --model_ckpt ./results/exp_unet_feature_visa/cashew/model_ema_latest.pth \
    --config_path ./results/exp_unet_feature_visa/cashew/config.yaml \
    --device cuda \
    --batch_size 1 \
    --save_all_images \

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit_feature.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 32 \
    --output_dir ./results/exp_unet_feature_visa/chewinggum \
    --model_ckpt ./results/exp_unet_feature_visa/chewinggum/model_ema_latest.pth \
    --config_path ./results/exp_unet_feature_visa/chewinggum/config.yaml \
    --device cuda \
    --batch_size 1 \
    --save_all_images \

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit_feature.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 32 \
    --output_dir ./results/exp_unet_feature_visa/fryum \
    --model_ckpt ./results/exp_unet_feature_visa/fryum/model_ema_latest.pth \
    --config_path ./results/exp_unet_feature_visa/fryum/config.yaml \
    --device cuda \
    --batch_size 1 \
    --save_all_images \

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit_feature.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 32 \
    --output_dir ./results/exp_unet_feature_visa/macaroni1 \
    --model_ckpt ./results/exp_unet_feature_visa/macaroni1/model_ema_latest.pth \
    --config_path ./results/exp_unet_feature_visa/macaroni1/config.yaml \
    --device cuda \
    --batch_size 1 \
    --save_all_images \

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit_feature.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 32 \
    --output_dir ./results/exp_unet_feature_visa/macaroni2 \
    --model_ckpt ./results/exp_unet_feature_visa/macaroni2/model_ema_latest.pth \
    --config_path ./results/exp_unet_feature_visa/macaroni2/config.yaml \
    --device cuda \
    --batch_size 1 \
    --save_all_images \

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit_feature.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 32 \
    --output_dir ./results/exp_unet_feature_visa/pcb1 \
    --model_ckpt ./results/exp_unet_feature_visa/pcb1/model_ema_latest.pth \
    --config_path ./results/exp_unet_feature_visa/pcb1/config.yaml \
    --device cuda \
    --batch_size 1 \
    --save_all_images \

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit_feature.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 32 \
    --output_dir ./results/exp_unet_feature_visa/pcb2 \
    --model_ckpt ./results/exp_unet_feature_visa/pcb2/model_ema_latest.pth \
    --config_path ./results/exp_unet_feature_visa/pcb2/config.yaml \
    --device cuda \
    --batch_size 1 \
    --save_all_images \

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit_feature.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 32 \
    --output_dir ./results/exp_unet_feature_visa/pcb3 \
    --model_ckpt ./results/exp_unet_feature_visa/pcb3/model_ema_latest.pth \
    --config_path ./results/exp_unet_feature_visa/pcb3/config.yaml \
    --device cuda \
    --batch_size 1 \
    --save_all_images \

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit_feature.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 32 \
    --output_dir ./results/exp_unet_feature_visa/pcb4 \
    --model_ckpt ./results/exp_unet_feature_visa/pcb4/model_ema_latest.pth \
    --config_path ./results/exp_unet_feature_visa/pcb4/config.yaml \
    --device cuda \
    --batch_size 1 \
    --save_all_images \

export CUDA_VISIBLE_DEVICES=0
python src/evaluate_dit_feature.py \
    --num_samples 1 \
    --num_inference_steps 100 \
    --recon_space feature \
    --start_step 32 \
    --output_dir ./results/exp_unet_feature_visa/pipe_fryum \
    --model_ckpt ./results/exp_unet_feature_visa/pipe_fryum/model_ema_latest.pth \
    --config_path ./results/exp_unet_feature_visa/pipe_fryum/config.yaml \
    --device cuda \
    --batch_size 1 \
    --save_all_images \
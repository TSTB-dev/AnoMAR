# bottle  capsule  grid      leather      metal_nut            pill        screw  toothbrush  wood
# cable   carpet   hazelnut  license.txt  mvtec_ad_evaluation  readme.txt  tile   transistor  zipper

export CUDA_VISIBLE_DEVICES=0
python3 ./src/train_dit.py --config_path ./configs/exp_ad/ad_dit_bottle.yml

# export CUDA_VISIBLE_DEVICES=1
# python3 ./src/train.py --config_path ./configs/exp_ad/ad_dit_cable.yml &

# export CUDA_VISIBLE_DEVICES=2
# python3 ./src/train.py --config_path ./configs/exp_ad/ad_dit_capsule.yml &

# export CUDA_VISIBLE_DEVICES=3
# python3 ./src/train.py --config_path ./configs/exp_ad/ad_dit_carpet.yml &

# export CUDA_VISIBLE_DEVICES=4
# python3 ./src/train.py --config_path ./configs/exp_ad/ad_dit_grid.yml &

# export CUDA_VISIBLE_DEVICES=5
# python3 ./src/train.py --config_path ./configs/exp_ad/ad_dit_hazelnut.yml &

# export CUDA_VISIBLE_DEVICES=6
# python3 ./src/train.py --config_path ./configs/exp_ad/ad_dit_leather.yml &

# export CUDA_VISIBLE_DEVICES=7
# python3 ./src/train.py --config_path ./configs/exp_ad/ad_dit_metal_nut.yml &

# wait 

# export CUDA_VISIBLE_DEVICES=0
# python3 ./src/train.py --config_path ./configs/exp_ad/ad_dit_pill.yml &

# export CUDA_VISIBLE_DEVICES=1
# python3 ./src/train.py --config_path ./configs/exp_ad/ad_dit_screw.yml &

# export CUDA_VISIBLE_DEVICES=2
# python3 ./src/train.py --config_path ./configs/exp_ad/ad_dit_tile.yml &

# export CUDA_VISIBLE_DEVICES=3
# python3 ./src/train.py --config_path ./configs/exp_ad/ad_dit_toothbrush.yml &

# export CUDA_VISIBLE_DEVICES=4
# python3 ./src/train.py --config_path ./configs/exp_ad/ad_dit_transistor.yml &

# export CUDA_VISIBLE_DEVICES=5
# python3 ./src/train.py --config_path ./configs/exp_ad/ad_dit_wood.yml &

# export CUDA_VISIBLE_DEVICES=6
# python3 ./src/train.py --config_path ./configs/exp_ad/ad_dit_zipper.yml &



# bottle  capsule  grid      leather      metal_nut            pill        screw  toothbrush  wood
# cable   carpet   hazelnut  license.txt  mvtec_ad_evaluation  readme.txt  tile   transistor  zipper



export CUDA_VISIBLE_DEVICES=0
python3 ./src/train_mim.py --config_path ./configs/exp_mim/ad_mim_bottle.yml &

export CUDA_VISIBLE_DEVICES=1
python3 ./src/train_mim.py --config_path ./configs/exp_mim/ad_mim_cable.yml &

export CUDA_VISIBLE_DEVICES=2
python3 ./src/train_mim.py --config_path ./configs/exp_mim/ad_mim_capsule.yml &

export CUDA_VISIBLE_DEVICES=3
python3 ./src/train_mim.py --config_path ./configs/exp_mim/ad_mim_carpet.yml &

export CUDA_VISIBLE_DEVICES=4
python3 ./src/train_mim.py --config_path ./configs/exp_mim/ad_mim_hazelnut.yml &

export CUDA_VISIBLE_DEVICES=5
python3 ./src/train_mim.py --config_path ./configs/exp_mim/ad_mim_leather.yml &


wait 

export CUDA_VISIBLE_DEVICES=1
python3 ./src/train_mim.py --config_path ./configs/exp_mim/ad_mim_pill.yml &

export CUDA_VISIBLE_DEVICES=2
python3 ./src/train_mim.py --config_path ./configs/exp_mim/ad_mim_screw.yml &

export CUDA_VISIBLE_DEVICES=3
python3 ./src/train_mim.py --config_path ./configs/exp_mim/ad_mim_tile.yml &

export CUDA_VISIBLE_DEVICES=4
python3 ./src/train_mim.py --config_path ./configs/exp_mim/ad_mim_transistor.yml &

export CUDA_VISIBLE_DEVICES=5
python3 ./src/train_mim.py --config_path ./configs/exp_mim/ad_mim_wood.yml &

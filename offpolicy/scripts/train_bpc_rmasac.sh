#!/bin/sh
env="BlueprintConstruction"
scenario_name="empty"
num_agents=2
num_boxes=4
floor_size=4.0
algo="rmasac"
exp="rebuttal-bpc"
seed_max=1

echo "env is ${env}, scenario name is ${scenario_name}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=7 python train/train_hns.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario_name} --num_agents ${num_agents} --num_boxes ${num_boxes} --floor_size ${floor_size} --seed ${seed} --episode_length 200 --actor_train_interval_step 1 --target_entropy_coef 0.3 --batch_size 256 --buffer_size 5000 --lr 5e-4 --tau 0.005 --num_env_steps 100000000 --eval_interval 50000 --num_eval_episodes 32
    echo "training is done!"
done

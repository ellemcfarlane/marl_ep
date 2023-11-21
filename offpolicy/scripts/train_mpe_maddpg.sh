#!/bin/sh
env="MPE"
scenario="simple_spread"
num_landmarks=3
num_agents=3
algo="maddpg"
exp="debug"
seed_max=1
# WARNING: make device num is correct, e.g. check avail with nvidid-smi
CUDA_VISIBLE_DEVICES=0
PYTHON_PATH=~/miniconda3/envs/qmix/bin/python3

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_max}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ${PYTHON_PATH} train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --n_rollout_threads 128 --episode_length 25 --actor_train_interval_step 1 --tau 0.005 --lr 7e-4 --num_env_steps 10000000 --batch_size 1000 --buffer_size 500000 --use_reward_normalization --use_wandb --user_name elles
    echo "training is done!"
done

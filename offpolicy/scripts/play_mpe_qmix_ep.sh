#!/bin/sh
env="MPE"
scenario="simple_spread" # aka cooperative navigation
num_landmarks=3
num_agents=3
algo="qmix_ep"
exp="${exp:-debug}" # default to experiment name "debug"
seed_max=1
# WARNING: make sure device num is correct, e.g. check avail with nvidid-smi
CUDA_VISIBLE_DEVICES=0
PYTHON_BIN=/work3/s222376/off-policy/env/bin/python3
PRETRAINED_QMIX=/work3/s222376/off-policy/offpolicy/models/epistemic_planner/
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_max}); do
    echo "seed is ${seed}:"
    echo "warning: make sure you're running on graphics terminal with vglrun for rendering"
    # TODO (elle): add epi_dir and model_dir flags
    # --cuda -> runs on CPU
    vglrun ${PYTHON_BIN} train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --episode_length 25 --batch_size 32 --tau 0.005 --lr 7e-4 --hard_update_interval_episode 100 --num_env_steps 10000000 --use_reward_normalization --user_name elles --model_dir ${PRETRAINED_QMIX} --play --cuda
    echo "training is done!"
done

USE_WANDB=false
EXP="qmix"

test:
	pytest tests/epistemic/test_marl_ep.py

train-mad:
	cd offpolicy/scripts/; \
	chmod +x ./train_mpe_maddpg.sh; \
	./train_mpe_maddpg.sh

train-ep:
	cd offpolicy/scripts/; \
	chmod +x ./train_mpe_qmix_ep.sh; \
	exp=${EXP} use_wandb=${USE_WANDB} ./train_mpe_qmix_ep.sh

play:
	cd offpolicy/scripts/; \
	chmod +x ./play_mpe_qmix.sh; \
	exp=${EXP} ./play_mpe_qmix.sh

play-ep:
	cd offpolicy/scripts/; \
	chmod +x ./play_mpe_qmix_ep.sh; \
	exp=${EXP} ./play_mpe_qmix_ep.sh

train:
	cd offpolicy/scripts/; \
	chmod +x ./train_mpe_qmix.sh; \
	exp=${EXP} use_wandb=${USE_WANDB} ./train_mpe_qmix.sh

train-hpc:
	cd offpolicy/scripts/; \
	chmod +x ./train_mpe_qmix.sh; \
	exp=${EXP} use_wandb=true ./train_mpe_qmix.sh

train-ep-hpc:
	cd offpolicy/scripts/; \
	chmod +x ./train_mpe_qmix_ep.sh; \
	exp=${EXP} use_wandb=true ./train_mpe_qmix_ep.sh

queue:
	bsub < jobscript.sh

# show's log outputs of latest hpc job
stat:
	@err_file=$$(ls -v gpu_*.err | tail -n 1); \
	out_file=$$(ls -v gpu_*.out | tail -n 1); \
	echo "Latest .err file: $$err_file"; \
	echo "Latest .out file: $$out_file"; \
	cat "$$err_file" "$$out_file"

profile:
	cd offpolicy/scripts/; \
	python3 read_profile.py

# /work3/s222376/off-policy/offpolicy/scripts/results/MPE/simple_spread/qmix/debug/wandb/run-20231123_143441-3g43g1v8/files/policy_0/q_network.pt
# /work3/s222376/off-policy/offpolicy/scripts/results/MPE/simple_spread/qmix/debug/wandb/run-20231123_143441-3g43g1v8/files/mixer.pt

# /work3/s222376/off-policy/offpolicy/scripts/results/MPE/simple_spread/qmix/debug/wandb/run-20231123_102645-18wnv6zj/files/policy_0/q_network.pt
# /work3/s222376/off-policy/offpolicy/scripts/results/MPE/simple_spread/qmix/debug/wandb/run-20231123_102645-18wnv6zj/files/mixer.pt
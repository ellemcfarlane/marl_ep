
train-mad:
	cd offpolicy/scripts/; \
	chmod +x ./train_mpe_maddpg.sh; \
	./train_mpe_maddpg.sh

train-ep:
	cd offpolicy/scripts/; \
	chmod +x ./train_mpe_qmix.sh; \
	./train_mpe_qmix_ep.sh

play-ep:
	cd offpolicy/scripts/; \
	chmod +x ./play_mpe_qmix_ep.sh; \
	./play_mpe_qmix_ep.sh

train:
	cd offpolicy/scripts/; \
	chmod +x ./train_mpe_qmix.sh; \
	./train_mpe_qmix.sh

train-hpc:
	cd offpolicy/scripts/; \
	chmod +x ./train_mpe_qmix.sh; \
	exp="qmix-hpc" ./train_mpe_qmix.sh

queue:
	bsub < jobscript.sh

# /work3/s222376/off-policy/offpolicy/scripts/results/MPE/simple_spread/qmix/debug/wandb/run-20231123_102645-18wnv6zj/files/policy_0/q_network.pt
# /work3/s222376/off-policy/offpolicy/scripts/results/MPE/simple_spread/qmix/debug/wandb/run-20231123_102645-18wnv6zj/files/mixer.pt
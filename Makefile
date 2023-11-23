
train-mad:
	cd offpolicy/scripts/; \
	chmod +x ./train_mpe_maddpg.sh; \
	./train_mpe_maddpg.sh

train-ep:
	cd offpolicy/scripts/; \
	chmod +x ./train_mpe_qmix.sh; \
	./train_mpe_qmix_ep.sh

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

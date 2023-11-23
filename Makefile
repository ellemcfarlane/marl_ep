
train-mad:
	cd offpolicy/scripts/; \
	chmod +x ./train_mpe_maddpg.sh; \
	./train_mpe_maddpg.sh

train:
	cd offpolicy/scripts/; \
	chmod +x ./train_mpe_qmix.sh; \
	./train_mpe_qmix.sh

queue:
	bsub < jobscript.sh

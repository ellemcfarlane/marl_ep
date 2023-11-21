
train:
	cd offpolicy/scripts/; \
	chmod +x ./train_mpe_maddpg.sh; \
	./train_mpe_maddpg.sh

queue:
	bsub < jobscript.sh

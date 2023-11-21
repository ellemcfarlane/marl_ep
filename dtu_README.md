
## 2. Installation on DTU HPC
* conda create -n qmix python=3.6
* conda activate qmix
* which python3 # double check points to python bin in conda env
* python3 pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
* module load cuda/10.1
* module load cudnn/v7.6.5.32-prod-cuda-10.1

```
# install offpolicy package
cd offpolicy
pip install -e .
```

### 2.2 Install MPE

``` Bash
# install this package first
pip install seaborn
```

There are 3 Cooperative scenarios in MPE:

* simple_spread
* simple_speaker_listener, which is 'Comm' scenario in paper
* simple_reference

## 3.Train
Here we use train_mpe_maddpg.sh as an example:
```
# 1. edit train_mpe_maddpg.sh env variables as you see fit - e.g. remove --use_wandb if you don't want to use wandb
# 1.2 if you do use wandb, make sure to change --user_name <username> to your own username in train_mpe_maddpg.sh
# 2. to run on gpu, use 'voltash' in dtu hpc"
make train
```
Local results are stored in subfold scripts/results. Note that we use Weights & Bias as the default visualization platform; to use Weights & Bias, please register and login to the platform first. More instructions for using Weights&Bias can be found in the official [documentation](https://docs.wandb.ai/). Adding the `--use_wandb` in command line or in the .sh file will use Tensorboard instead of Weights & Biases. 

## 4. Results
Results for the performance of RMADDPG and QMIX on the Particle Envs and QMIX in SMAC are depicted [here](https://docs.google.com/document/d/1s0Kb76b7v4WGyhiCNLrt9St-WvhGnl2AUQCe1FS-ADM/edit?usp=sharing). These results are obtained using a normal (not prioitized) replay buffer.

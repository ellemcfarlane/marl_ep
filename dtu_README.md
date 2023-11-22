
## 2. Installation on DTU HPC
* conda create -n qmix python=3.6 OR if you don't have space on home dir and have a scratch dir, please see section 'DTU HPC more info' to see how to run conda create command
* conda activate qmix
* which python3 # double check points to python bin in conda env
* module load cuda/10.1 # you must run these icuda commands before installing torch otherwise it will say version not found!!
* module load cudnn/v7.6.5.32-prod-cuda-10.1
* python3 -m pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
* python3 -m pip install -r dtu_requirements.txt

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
# 3. if run into dependency issues, python3 -m pip install <missing> according to version listed in dtu_requirements.txt
```
Local results are stored in subfold scripts/results. Note that we use Weights & Bias as the default visualization platform; to use Weights & Bias, please register and login to the platform first. More instructions for using Weights&Bias can be found in the official [documentation](https://docs.wandb.ai/). Adding the `--use_wandb` in command line or in the .sh file will use Tensorboard instead of Weights & Biases. 

### 3.1 Train on DTU HPC
* edit email in jobscript.sh to be your own (else: spam me)
* `make queue` to submit job to queue
* bstat to monitor job status
* look at job output file via 'ls' in folder you submitted job from and see 'gpu_*.out' and 'gpu_*.err' files
* see wandb output e.g. at https://wandb.ai/elles/MPE/runs/z4277c1c?workspace=user-ellesummer
```
## 4. Results
Results for the performance of RMADDPG and QMIX on the Particle Envs and QMIX in SMAC are depicted [here](https://docs.google.com/document/d/1s0Kb76b7v4WGyhiCNLrt9St-WvhGnl2AUQCe1FS-ADM/edit?usp=sharing). These results are obtained using a normal (not prioitized) replay buffer.

### 5. DTU HPC more info
https://docs.google.com/document/d/1pBBmoLTj_JPWiCSFYzfHj646bb8uUCh8lMetJxnE68c/edit
https://skaftenicki.github.io/dtu_mlops/s10_extra/high_performance_clusters/

#### 5.1 conda create on scratch space directory
If running into python binary issues with conda in your scratch space (aka when using --prefix to point to scratch), make sure to:
s222376@n-62-20-1 /work3/s222376 $ conda config --set always_copy True
s222376@n-62-20-1 /work3/s222376 $ conda config --show | grep always_copy
always_copy: True
s222376@n-62-20-1 /work3/s222376 $ conda create --prefix=<scratch-dir>/off-policy/env python=3.6

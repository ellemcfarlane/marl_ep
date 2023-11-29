
This repo attempts to reproduce the results in the paper [Multi-Agent Reinforcement Learning with Epistemic Priors](https://prl-theworkshop.github.io/prl2023-icaps/papers/multi-agent-reinforcement-learning.pdf) by Walker et al. (2023) and is almost completely based on [Off-Policy Multi-Agent Reinforcement Learning (MARL) Algorithms](https://github.com/marlbenchmark/off-policy) with changes for epistemic learning as described in the paper.

## 1. Usage
For original usage, please see [original_README.md](original_README.md).

Otherwise, for qmix training with epistemic priors:
* locally: `make train-ep`
* DTU HPC: `make train-ep-hpc`

For normal qmix training:
* locally: `make train`
* DTU HPC: `make train-hpc`

To modify training parameters, please see [train_mpe_qmix_ep.sh](offpolicy/scripts/train_mpe_qmix_ep.sh) and [train_mpe_qmix.sh](offpolicy/scripts/train_mpe_qmix.sh).

"Playing"/visualization of random MPE Spread scenario:
* with priors: `make play-ep`, edit the MODEL_DIR var in [play_mpe_qmix_ep.sh](offpolicy/scripts/play_mpe_qmix_ep.sh) to point to the model you want to play.
* NOTE: remove the vglrun command from make play-ep in Makefile if you are not on a compatible system

## 2. Models
* see our trained epistemic model in offpolicy/models/epistemic_planner
* see our models trained with the epistemic planner in offpolicy/models/qmix_ep/*

## 3. Installation on DTU HPC
* conda create -n qmix python=3.6 OR if you don't have space on home dir and have a scratch dir, please see section 'DTU HPC more info' to see how to run conda create command
* conda activate qmix
* which python3 # double check points to python bin in conda env
* module load cuda/10.1 # you must run these icuda commands before installing torch otherwise it will say version not found!!
* module load cudnn/v7.6.5.32-prod-cuda-10.1
* python3 -m pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
* python3 -m pip install -r requirements.txt

```
# install offpolicy package
cd marl_ep
python3 -m pip install -e .
```

### 3.2 Install MPE

``` Bash
# install this package first
python3 -m pip install seaborn
```

There are 3 Cooperative scenarios in MPE:

* simple_spread
* simple_speaker_listener, which is 'Comm' scenario in paper
* simple_reference

## 4. Results
<TODO: link to our overleaf paper with results>

## 5. Training on DTU HPC
* edit email in jobscript.sh to be your own (else: spam me)
* `make queue` to submit job to queue
* `make stat` to monitor job status
* see wandb output e.g. at https://wandb.ai/elles/MPE/runs/z4277c1c?workspace=user-ellesummer

### 5.1. DTU HPC more info
https://docs.google.com/document/d/1pBBmoLTj_JPWiCSFYzfHj646bb8uUCh8lMetJxnE68c/edit
https://skaftenicki.github.io/dtu_mlops/s10_extra/high_performance_clusters/

### 5.2 conda create on scratch space directory
If running into python binary issues with conda in your scratch space (aka when using --prefix to point to scratch), make sure to:
s222376@n-62-20-1 /work3/s222376 $ conda config --set always_copy True
s222376@n-62-20-1 /work3/s222376 $ conda config --show | grep always_copy
always_copy: True
s222376@n-62-20-1 /work3/s222376 $ conda create --prefix=<scratch-dir>/off-policy/env python=3.6

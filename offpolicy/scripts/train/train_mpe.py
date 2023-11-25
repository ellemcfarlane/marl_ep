import sys
import os
import numpy as np
from pathlib import Path
import socket
import wandb
import setproctitle
import torch
from offpolicy.config import get_config
from offpolicy.utils.util import get_cent_act_dim, get_dim_from_space
from offpolicy.envs.mpe.MPE_Env import MPEEnv
from offpolicy.envs.env_wrappers import DummyVecEnv, SubprocVecEnv
from copy import deepcopy

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=3, help="number of agents")
    parser.add_argument('--use_same_share_obs', action='store_false',
                        default=True, help="Whether to use available actions")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    print(f"all_args: {all_args}")
    print(f"cuda avail? {torch.cuda.is_available()}")
    if all_args.dry_run:
        exit()
    # cuda and # threads
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # setup file to output tensorboard, hyperparameters, and saved models
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name

    epi_run_dir = Path(f"{run_dir}_epi")
    
    
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    if not epi_run_dir.exists():
        os.makedirs(str(epi_run_dir))

    if all_args.use_wandb:
        # init wandb
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
        print("wandb init done")
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[
                                 1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(
        all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # set seeds
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # create env
    env = make_train_env(all_args)
    num_agents = all_args.num_agents

    # create policies and mapping fn
    if all_args.share_policy:
        policy_info = {
            'policy_0': {"cent_obs_dim": get_dim_from_space(env.share_observation_space[0]),
                         "cent_act_dim": get_cent_act_dim(env.action_space),
                         "obs_space": env.observation_space[0],
                         "share_obs_space": env.share_observation_space[0],
                         "act_space": env.action_space[0]}
        }

        def policy_mapping_fn(id): return 'policy_0'
    else:
        policy_info = {
            'policy_' + str(agent_id): {"cent_obs_dim": get_dim_from_space(env.share_observation_space[agent_id]),
                                        "cent_act_dim": get_cent_act_dim(env.action_space),
                                        "obs_space": env.observation_space[agent_id],
                                        "share_obs_space": env.share_observation_space[agent_id],
                                        "act_space": env.action_space[agent_id]}
            for agent_id in range(num_agents)
        }

        def policy_mapping_fn(agent_id): return 'policy_' + str(agent_id)

    # choose algo
    if all_args.algorithm_name in ["rmatd3", "rmaddpg", "rmasac", "qmix", "qmix_ep", "vdn"]:
        from offpolicy.runner.rnn.mpe_runner import MPERunner as Runner
        # TODO (elle): why only support 1 env in recurrent version?
        assert all_args.n_rollout_threads == 1, (
            "only support 1 env in recurrent version.")
        eval_env = env
    elif all_args.algorithm_name in ["matd3", "maddpg", "masac", "mqmix", "mvdn"]:
        from offpolicy.runner.mlp.mpe_runner import MPERunner as Runner
        eval_env = make_eval_env(all_args)
    else:
        raise NotImplementedError

    # load pretrained QMIX as epistemic planner
    # total_num_steps = 0
    if all_args.algorithm_name in ["qmix_ep"]:
        assert all_args.epi_dir is not None, "Must specify epi_dir for epistemic planner"
        assert all_args.model_dir is None, "Must not specify model_dir for epistemic planner"
        epi_args = deepcopy(all_args)
        epi_args.buffer_size = 1 # only want one plan for given env
        ep_planner_config = {"args": epi_args,
                "policy_info": policy_info,
                "policy_mapping_fn": policy_mapping_fn,
                "env": deepcopy(env), # TODO (elle): double check copies correctly, else use make_train_env(all_args)
                "eval_env": make_eval_env(epi_args),
                "num_agents": num_agents,
                "device": device,
                "use_same_share_obs": True,
                "run_dir": epi_run_dir,
        }
        epistemic_planner = Runner(config=ep_planner_config)

    config = {"args": all_args,
              "policy_info": policy_info,
              "policy_mapping_fn": policy_mapping_fn,
              "env": env,
              "eval_env": make_eval_env(all_args), # TODO fix to generalize to more than qmix and qmix_ep i.e. eval_env = <>
              "num_agents": num_agents,
              "device": device,
              "use_same_share_obs": all_args.use_same_share_obs,
              "run_dir": run_dir,
              "epistemic_planner": epistemic_planner if all_args.algorithm_name in ["qmix_ep"] else None,
    }

    total_num_steps = 0
    runner = Runner(config=config)
    print("running?") 
    if not all_args.play:
        while total_num_steps < all_args.num_env_steps:
            total_num_steps = runner.run()
    else:
        runner.play()
    env.close()
    if all_args.use_eval and (eval_env is not env):
        eval_env.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(
            str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])

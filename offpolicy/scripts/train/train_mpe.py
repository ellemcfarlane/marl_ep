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
from offpolicy.utils.util import setup_logging
import logging

setup_logging()

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

# modifies args
def add_epistem_prior_dims(policy_info, args) -> None:
    epi_dims = (args.num_agents - 1) * 2
    if args.share_policy:
        policy_info['policy_0']['cent_obs_dim'] += epi_dims * args.num_agents
        policy_info['policy_0']['obs_space'] += epi_dims
    else:
        for agent_id in range(args.num_agents):
            policy_info['policy_' + str(agent_id)]['cent_obs_dim'] += epi_dims * args.num_agents
            policy_info['policy_' + str(agent_id)]['obs_space'] += epi_dims

def get_policy_info_from_env(env, args):
    if args.share_policy:
        policy_info = {
            'policy_0': {"cent_obs_dim": get_dim_from_space(env.share_observation_space[0]),
                         "cent_act_dim": get_cent_act_dim(env.action_space),
                         "obs_space": env.observation_space[0],
                         "share_obs_space": env.share_observation_space[0],
                         "act_space": env.action_space[0]}
        }
    else:
        policy_info = {
            'policy_' + str(agent_id): {"cent_obs_dim": get_dim_from_space(env.share_observation_space[agent_id]),
                                        "cent_act_dim": get_cent_act_dim(env.action_space),
                                        "obs_space": env.observation_space[agent_id],
                                        "share_obs_space": env.share_observation_space[agent_id],
                                        "act_space": env.action_space[agent_id]}
            for agent_id in range(args.num_agents)
        }

    return policy_info

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    logging.info(f"all_args: {all_args}")
    logging.info(f"cuda avail? {torch.cuda.is_available()}")
    if all_args.dry_run:
        exit()
    # cuda and # threads
    if all_args.cuda and torch.cuda.is_available():
        logging.info("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        logging.info("choose to use cpu...")
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
    policy_info = get_policy_info_from_env(env, all_args)
    if all_args.share_policy:
        def policy_mapping_fn(id): return 'policy_0'
    else:
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

    # total_num_steps = 0
    if all_args.epistemic and all_args.algorithm_name in ["qmix_ep"]:
        # load pretrained QMIX as epistemic planner
        assert all_args.epi_dir is not None, "Must specify epi_dir for epistemic planner"
        epi_args = deepcopy(all_args)
        epi_args.model_dir = all_args.epi_dir
        epi_args.buffer_size = 1 # only want one plan for given env
        # config for model that was not trained with priors but that will be used to get the priors, i.e. serve as epistemic planner
        epi_args.epistemic = False # this is pretrained model, so not don't want to use epistemic priors
        logging.info(f"epistemic planner args: {epi_args}")
        epi_env = make_train_env(epi_args)
        epi_eval_env = make_eval_env(epi_args)
        epi_policy_info = get_policy_info_from_env(epi_env, epi_args)

        init_epi_agent_poses = np.array([epi_env.envs[0].world.agents[i].state.p_pos for i in range(all_args.num_agents)])
        init_agent_poses = np.array([env.envs[0].world.agents[i].state.p_pos for i in range(all_args.num_agents)])
        init_epi_landmark_poses = np.array([epi_env.envs[0].world.landmarks[i].state.p_pos for i in range(all_args.num_agents)])
        init_landmark_poses = np.array([env.envs[0].world.landmarks[i].state.p_pos for i in range(all_args.num_agents)])

        if not np.all(init_epi_agent_poses == init_agent_poses) or not np.all(init_epi_landmark_poses == init_landmark_poses):
            assert np.all(init_epi_agent_poses == init_agent_poses), f"init_epi_agent_poses: {init_epi_agent_poses} don't match init_agent_poses: {init_agent_poses}"
            assert np.all(init_epi_landmark_poses == init_landmark_poses), f"init_epi_landmark_poses: {init_epi_landmark_poses} don't match init_landmark_poses: {init_landmark_poses}"
        else:
            logging.info("congrats!!! init conds match between epi env and normal env :D")

        ep_planner_config = {
            "args": epi_args,
            "policy_info": epi_policy_info,
            "policy_mapping_fn": policy_mapping_fn, # TODO (elle): will differ for epistemic planner (centralised, so 1 policy) vs qmix (decentralised, so num_agents policies)
            "env": epi_env, # TODO (elle): double check copies correctly, else use make_train_env(all_args)
            "eval_env": epi_eval_env,
            "num_agents": num_agents,
            "device": device,
            "use_same_share_obs": True,
            "run_dir": epi_run_dir,
            "use_epi_priors": False,
            "skip_warmup": True,
        }
        epistemic_planner = Runner(config=ep_planner_config)
        # epistemic_planner = "dummy planner"
        # config for model that will now be trained with priors
        assert all_args.model_dir is None, "Must not specify model_dir if using epistemic planner"
        # update policy dimensions to include epistemic prior dims
        logging.info(f"training qmix args: {all_args}")
        config = {
            "args": all_args,
            "policy_info": policy_info,
            "policy_mapping_fn": policy_mapping_fn,
            "env": env,
            "eval_env": make_eval_env(all_args), # TODO fix to generalize to more than qmix and qmix_ep i.e. eval_env = <>
            "num_agents": num_agents,
            "device": device,
            "use_same_share_obs": all_args.use_same_share_obs, # TODO: set false!
            "run_dir": run_dir,
            "epistemic_planner": epistemic_planner,
            "use_epi_priors": True,
        }
    else:
        config = {
            "args": all_args,
            "policy_info": policy_info,
            "policy_mapping_fn": policy_mapping_fn,
            "env": env,
            "eval_env": eval_env,
            "num_agents": num_agents,
            "device": device,
            "use_same_share_obs": all_args.use_same_share_obs,
            "run_dir": run_dir
        }

    config["skip_warmup"] = True if all_args.play else False
    if all_args.play and all_args.model_dir is None:
        raise ValueError("Must specify model_dir if playing")
    elif all_args.epi_dir is not None:
        logging.warning("NO EPI_DIR MODEL SPECIFIED, PLAYING WITHOUT PRIORS")
    total_num_steps = 0
    runner = Runner(config=config)
    logging.info("running?")
    episodes = 0
    if not all_args.play:
        while total_num_steps < all_args.num_env_steps:
            logging.debug("calling runner.run()")
            total_num_steps = runner.run()
            logging.debug(".run() done")
            logging.debug(f"episode {episodes} complete, total_num_steps {total_num_steps}")
            episodes += 1
    else:
        # add "skip_warmup": True to config
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

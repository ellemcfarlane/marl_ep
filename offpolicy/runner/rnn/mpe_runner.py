import numpy as np
import torch
import time
import sys
from offpolicy.runner.rnn.base_runner import RecRunner
from offpolicy.utils.util import setup_logging
import logging
from copy import deepcopy

setup_logging()

class MPERunner(RecRunner):
    """Runner class for Multiagent Particle Envs (MPE). See parent class for more information."""
    def __init__(self, config):
        super(MPERunner, self).__init__(config)
        self.collecter = self.shared_collect_rollout if self.share_policy else self.separated_collect_rollout
        # fill replay buffer with random actions
        if not self.skip_warmup:
            num_warmup_episodes = max((self.batch_size, self.args.num_random_episodes))
            logging.debug("mperunner.__init__.warmup")
            logging.info(f"####### COLLECTING {num_warmup_episodes} RANDOM EPISODES AS WARMUP #######")
            self.collect_random_episodes(num_warmup_episodes)
        else:
            logging.info("####### SKIPPING WARMUP #######")
        self.start = time.time()
        self.log_clear()
    
    def eval(self):
        """Collect episodes to evaluate the policy."""
        logging.debug("mperunner.eval.prep_rollout")
        self.trainer.prep_rollout()
        eval_infos = {}
        eval_infos['average_episode_rewards'] = []

        for _ in range(self.args.num_eval_episodes):
            env_info = self.collecter( explore=False, training_episode=False, warmup=False)
            for k, v in env_info.items():
                eval_infos[k].append(v)

        self.log_env(eval_infos, suffix="eval_")
    
    def play(self):
        """Play an episode using the trained policy."""
        env_info = {}
        # only 1 policy since all agents share weights
        p_id = "policy_0"
        policy = self.policies[p_id]

        env = self.eval_env

        obs = env.reset()

        rnn_states_batch = np.zeros((self.num_envs * self.num_agents, self.hidden_size), dtype=np.float32)
        last_acts_batch = np.zeros((self.num_envs * self.num_agents, policy.output_dim), dtype=np.float32)

        # initialize variables to store episode information.
        episode_obs = {p_id : np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_share_obs = {p_id : np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.central_obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_acts = {p_id : np.zeros((self.episode_length, self.num_envs, self.num_agents, policy.output_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_rewards = {p_id : np.zeros((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones = {p_id : np.ones((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones_env = {p_id : np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32) for p_id in self.policy_ids}

        t = 0
        logging.info(f"playing episode of length {self.episode_length}")
        while t < self.episode_length:
            share_obs = obs.reshape(self.num_envs, -1)
            # group observations from parallel envs into one batch to process at once
            obs_batch = np.concatenate(obs)
            # get actions for all agents to step the env

            acts_batch, rnn_states_batch, _ = policy.get_actions(obs_batch,
                                                    last_acts_batch,
                                                    rnn_states_batch,
                                                    explore=False)

            acts_batch = acts_batch if isinstance(acts_batch, np.ndarray) else acts_batch.cpu().detach().numpy()
            # update rnn hidden state
            rnn_states_batch = rnn_states_batch if isinstance(rnn_states_batch, np.ndarray) else rnn_states_batch.cpu().detach().numpy()
            last_acts_batch = acts_batch

            env_acts = np.split(acts_batch, self.num_envs)
            # env step and store the relevant episode informatio
            next_obs, rewards, dones, infos = env.step(env_acts)
            env.render()
            # sleep to slow down rendering
            time.sleep(0.1)

            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(dones_env) or t == self.episode_length - 1

            episode_obs[p_id][t] = obs
            episode_share_obs[p_id][t] = share_obs
            episode_acts[p_id][t] = np.stack(env_acts)
            episode_rewards[p_id][t] = rewards
            episode_dones[p_id][t] = dones
            episode_dones_env[p_id][t] = dones_env
            t += 1

            obs = next_obs

            if terminate_episodes:
                break

        episode_obs[p_id][t] = obs
        episode_share_obs[p_id][t] = obs.reshape(self.num_envs, -1)

        average_episode_rewards = np.mean(np.sum(episode_rewards[p_id], axis=0))
        env_info['average_episode_rewards'] = average_episode_rewards
        logging.info(f"average_episode_rewards: {average_episode_rewards}")

        return env_info
    
    @staticmethod
    def get_epistemic_priors(agent_pos, other_poses):
        """
        Get the relative positions of other agents to the given agent.
        :param agent_pos: (np.ndarray) position of the given agent.
        :param other_poses: (np.ndarray) positions of other agents.
        :return relative_pos: (np.ndarray) relative positions of other agents to the given agent.
        """
        epistemic_priors = []
        for pos in other_poses:
            # TODO (elle): discuss with team about whether to include self in epistemic priors
            # if other_agent_id == agent_id:
            #     continue
            assert pos.shape == (2,), f"pos.shape: {pos.shape}; should be (2,)"
            assert agent_pos.shape == (2,), f"agent_pos.shape: {agent_pos.shape}; should be (2,)"
            relative_pos = pos - agent_pos
            epistemic_priors.append(relative_pos)
        return np.array(epistemic_priors)
    
    @staticmethod
    def agent_pos_from_joint_obs(agent_id, obss):
        """
        Get the agent's position from the joint-observation.
        :param agent_id: (int) agent id to get the position of.
        :param obss: (np.ndarray) observation of shape (n_envs, n_agents, 18) to get the agent's position from - so dim of each agent's individual obs is 18
        :return agent_pos: (np.ndarray) agent's position.
        """
        env_idx = 0
        local_obs = obss[env_idx, agent_id]
        return MPERunner.agent_pos_from_obs(local_obs)

    def agent_pos_from_obs(local_obs):
        """
        Get the agent's position from the observation.
        :param agent_id: (int) agent id to get the position of.
        :param local_obs: (np.ndarray) one agent's observation of shape (18,1) to get the agent's position from - so dim of each agent's individual obs is 18
        :return agent_pos: (np.ndarray) agent's position.
        """
        agent_pos = local_obs[2:4]
        return agent_pos
        
    @staticmethod
    def add_priors_to_obs(obs, agent_poses_in_plan):
        """
        Add priors to the observation.
        :param obs: (np.ndarray) observation of shape (n_envs, n_agents, 18) to add priors to - so dim of each agent's individual obs is 18
        :param priors: (np.ndarray) priors to add to the observation.
        :return obs: (np.ndarray) observation with priors added, with shape (n_envs, n_agents, 18 + 2*(n_agents))
        """
        env_idx = 0 # only one env for now
        # make modified_obs with same shape as obs but with priors added
        num_envs, num_agents, obs_dim = obs.shape
        modified_obs = np.zeros((num_envs, num_agents, obs_dim + 2 * num_agents))
        # shape is (2 + 2 + num_landmarks * 2 + (num_agents-1) * 2 + (num_agents-1) * 2) = 4 + 6 + 4 + 4 = 18 when num_agents = 3, num_landmarks = 3
        current_agent_poses = [MPERunner.agent_pos_from_joint_obs(i, obs) for i in range(num_agents)]
        for agent_id, agent_pos in enumerate(current_agent_poses):
            assert agent_poses_in_plan.shape == (num_agents, 2), f"agent_poses_in_plan.shape: {agent_poses_in_plan.shape}; should be {(num_agents, 2)}"
            priors = MPERunner.get_epistemic_priors(agent_pos, agent_poses_in_plan)
            exp_priors_shape = (num_agents, 2)
            assert priors.shape == exp_priors_shape, f"priors.shape: {priors.shape}; should be {exp_priors_shape}"
            logging.debug(f"priors: {priors}")
            # now flattern priors to add to obs
            priors = priors.flatten()
            logging.debug(f"priors after flattening: {priors}")
            assert priors.shape == (2*(num_agents),), f"priors.shape: {priors.shape} after flattening, should be {(2*(num_agents),)}"
            old_entry = obs[env_idx, agent_id]
            logging.debug(f"old_entry.shape: {old_entry.shape}")
            logging.debug(f"priors.shape: {priors.shape}")
            new_entry = np.concatenate((old_entry, priors), axis=-1)
            logging.debug(f"new_entry.shape: {new_entry.shape}")
            assert new_entry.shape == (obs_dim + 2*(num_agents),), f"new_entry.shape: {new_entry.shape}; should be {(obs_dim + 2*(num_agents),)}"
            modified_obs[env_idx, agent_id] = new_entry
        return modified_obs

    @staticmethod
    # gets poses from epistemic plan
    def joint_pos_in_epistemic_plan(rollout_obss, t, num_agents):
        """
        Get the agent poses in the rollout at the given time step.
        :param rollout_obss: (ndarray) of (ep_len + 1, n_envs, n_agents, obs_dim); NOTE: is (n_agents, ep_len + 1, n_envs, obs_dim) if grabbed from buffer
        :param t: (int) time step to get the agent poses at.
        :return agent_poses: (np.ndarray) agent poses at the given time step.
        """
        agent_poses = []
        env_idx = 0 # only one env for now
        for agent_id in range(num_agents):
            agent_obs_at_t = rollout_obss[t][env_idx][agent_id]
            # assert agent_obs_at_t.shape == (18,), f"agent_obs_at_t.shape: {agent_obs_at_t.shape}; should be (18,)"
            # obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
            # shape is (2 + 2 + num_landmarks * 2 + (num_agents-1) * 2 + (num_agents-1) * 2) = 4 + 6 + 4 + 4 = 18 when num_agents = 3, num_landmarks = 3
            agent_pos = MPERunner.agent_pos_from_obs(agent_obs_at_t)
            agent_poses.append(agent_pos)
        return np.array(agent_poses)

    @staticmethod
    @torch.no_grad()
    # TODO: make self method and just call epistemic_planner.collect_epistemic_plan()
    def collect_epistemic_plan(epistemic_planner, env, render=False):
        env_info = {}
        # only 1 policy since all agents share weights
        p_id = "policy_0"
        policy = epistemic_planner.policies[p_id]

        obs = env.reset()

        rnn_states_batch = np.zeros((epistemic_planner.num_envs * epistemic_planner.num_agents, epistemic_planner.hidden_size), dtype=np.float32)
        last_acts_batch = np.zeros((epistemic_planner.num_envs * epistemic_planner.num_agents, policy.output_dim), dtype=np.float32)

        # initialize variables to store episode information.
        # (n_agents, ep_len + 1, n_envs, obs_dim)
        episode_obs = {p_id : np.zeros((epistemic_planner.episode_length + 1, epistemic_planner.num_envs, epistemic_planner.num_agents, policy.obs_dim), dtype=np.float32) for p_id in epistemic_planner.policy_ids}
        episode_share_obs = {p_id : np.zeros((epistemic_planner.episode_length + 1, epistemic_planner.num_envs, epistemic_planner.num_agents, policy.central_obs_dim), dtype=np.float32) for p_id in epistemic_planner.policy_ids}
        episode_acts = {p_id : np.zeros((epistemic_planner.episode_length, epistemic_planner.num_envs, epistemic_planner.num_agents, policy.output_dim), dtype=np.float32) for p_id in epistemic_planner.policy_ids}
        episode_rewards = {p_id : np.zeros((epistemic_planner.episode_length, epistemic_planner.num_envs, epistemic_planner.num_agents, 1), dtype=np.float32) for p_id in epistemic_planner.policy_ids}
        episode_dones = {p_id : np.ones((epistemic_planner.episode_length, epistemic_planner.num_envs, epistemic_planner.num_agents, 1), dtype=np.float32) for p_id in epistemic_planner.policy_ids}
        episode_dones_env = {p_id : np.ones((epistemic_planner.episode_length, epistemic_planner.num_envs, 1), dtype=np.float32) for p_id in epistemic_planner.policy_ids}
        episode_avail_acts = {p_id : None for p_id in epistemic_planner.policy_ids}

        t = 0
        while t < epistemic_planner.episode_length:
            share_obs = obs.reshape(epistemic_planner.num_envs, -1)
            # group observations from parallel envs into one batch to process at once
            obs_batch = np.concatenate(obs)
            # get actions for all agents to step the env
            acts_batch, rnn_states_batch, _ = policy.get_actions(obs_batch,
                                                    last_acts_batch,
                                                    rnn_states_batch,
                                                    explore=False)
            acts_batch = acts_batch if isinstance(acts_batch, np.ndarray) else acts_batch.cpu().detach().numpy()
            # update rnn hidden state
            rnn_states_batch = rnn_states_batch if isinstance(rnn_states_batch, np.ndarray) else rnn_states_batch.cpu().detach().numpy()
            last_acts_batch = acts_batch

            env_acts = np.split(acts_batch, epistemic_planner.num_envs)
            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)
            env.render()
            # sleep to slow down rendering
            time.sleep(0.1)
            epistemic_planner.total_env_steps += epistemic_planner.num_envs

            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(dones_env) or t == epistemic_planner.episode_length - 1

            episode_obs[p_id][t] = obs
            episode_share_obs[p_id][t] = share_obs
            episode_acts[p_id][t] = np.stack(env_acts)
            episode_rewards[p_id][t] = rewards
            episode_dones[p_id][t] = dones
            episode_dones_env[p_id][t] = dones_env
            t += 1

            obs = next_obs

            if terminate_episodes:
                break

        episode_obs[p_id][t] = obs
        episode_share_obs[p_id][t] = obs.reshape(epistemic_planner.num_envs, -1)

        average_episode_rewards = np.mean(np.sum(episode_rewards[p_id], axis=0))
        env_info['average_episode_rewards'] = average_episode_rewards
        plan = episode_obs
        return plan, env_info


    # for mpe-simple_spread and mpe-simple_reference  
    @torch.no_grad() 
    def shared_collect_rollout(self, explore=True, training_episode=True, warmup=False, render=False):
        """
        Collect a rollout and store it in the buffer. All agents share a single policy.
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        logging.debug("mperunner.shared_collect_rollout")
        env_info = {}
        # only 1 policy since all agents share weights
        p_id = "policy_0"
        policy = self.policies[p_id]

        env = self.env if training_episode or warmup else self.eval_env
        obs = env.reset()
        logging.debug(f'obs.shape at env.reset(): {obs.shape}')
        rnn_states_batch = np.zeros((self.num_envs * self.num_agents, self.hidden_size), dtype=np.float32)
        last_acts_batch = np.zeros((self.num_envs * self.num_agents, policy.output_dim), dtype=np.float32)

        # initialize variables to store episode information.
        episode_obs = {p_id : np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_share_obs = {p_id : np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.central_obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_acts = {p_id : np.zeros((self.episode_length, self.num_envs, self.num_agents, policy.output_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_rewards = {p_id : np.zeros((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones = {p_id : np.ones((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones_env = {p_id : np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_avail_acts = {p_id : None for p_id in self.policy_ids}

        logging.debug(f"episode obs shape: {episode_obs[p_id].shape}")
        t = 0
        if self.epistemic_planner is not None:
            logging.debug(f"COLLECTING PRIORS")
            assert self.epistemic_planner.epistemic_planner is None, f"epistemic planner's epistemic planner should be None, not {self.epistemic_planner.epistemic_planner}"
            # copy current env, set it as epistemic_planner's env and eval_env
            # call planner to collect rollout of ONE episode but with same steps as current & store in buffer
            # extract plan from buffer and set as epistemic plan or whatever - make sure doesn't affect loss fxn stuff
            epi_env = deepcopy(env)
            init_epi_agent_poses = np.array([epi_env.envs[0].world.agents[i].state.p_pos for i in range(self.num_agents)])
            init_epi_landmark_poses = np.array([epi_env.envs[0].world.landmarks[i].state.p_pos for i in range(self.num_agents)])
            init_agent_poses = np.array([env.envs[0].world.agents[i].state.p_pos for i in range(self.num_agents)])
            init_landmark_poses = np.array([env.envs[0].world.landmarks[i].state.p_pos for i in range(self.num_agents)])
            # make sure epistemic planner is solving the same
            if not np.all(init_epi_agent_poses == init_agent_poses):
                assert np.all(init_epi_agent_poses == init_agent_poses), f"init_epi_agent_poses: {init_epi_agent_poses} don't match init_agent_poses: {init_agent_poses}"
                assert np.all(init_epi_landmark_poses == init_landmark_poses), f"init_epi_landmark_poses: {init_epi_landmark_poses} don't match init_landmark_poses: {init_landmark_poses}"
            # plan has dims (epistemic_planner.episode_length + 1, epistemic_planner.num_envs, epistemic_planner.num_agents, policy.obs_dim)
            plan, env_info = MPERunner.collect_epistemic_plan(self.epistemic_planner, epi_env)
            agent_rollouts_obs_comp = plan[p_id]
            logging.info(f"epistemic planner collected plan of len {agent_rollouts_obs_comp.shape[0]} with reward {env_info['average_episode_rewards']}")
            # logging.debug(f"plan's obs {agent_rollouts_obs_comp.shape}, ep_len {self.episode_length}")
            # get types of each too
        while t < self.episode_length:
            if self.epistemic_planner is not None:
                logging.debug(f"BEFORE ADDING PRIORS obs: {obs.shape}, n_envs {self.num_envs}, n_agents {self.num_agents}")
                agent_poses_in_plan_at_time = self.joint_pos_in_epistemic_plan(agent_rollouts_obs_comp, t, self.num_agents)
                # agent_poses_in_plan_at_time = np.array([[0,0], [0,0], [0,0]])
                obs = MPERunner.add_priors_to_obs(obs, agent_poses_in_plan_at_time)
                logging.debug(f"AFTER ADDING PRIORS obs: {obs.shape}, n_envs {self.num_envs}, n_agents {self.num_agents}")
            share_obs = obs.reshape(self.num_envs, -1)
            # group observations from parallel envs into one batch to process at once
            obs_batch = np.concatenate(obs)
            # get actions for all agents to step the env
            if warmup:
                # completely random actions in pre-training warmup phase
                acts_batch = policy.get_random_actions(obs_batch)
                # get new rnn hidden state
                _, rnn_states_batch, _ = policy.get_actions(obs_batch,
                                                            last_acts_batch,
                                                            rnn_states_batch)
            else:
                # get actions with exploration noise (eps-greedy/Gaussian)
                acts_batch, rnn_states_batch, _ = policy.get_actions(obs_batch,
                                                                    last_acts_batch,
                                                                    rnn_states_batch,
                                                                    t_env=self.total_env_steps,
                                                                    explore=explore)
            acts_batch = acts_batch if isinstance(acts_batch, np.ndarray) else acts_batch.cpu().detach().numpy()
            # update rnn hidden state
            rnn_states_batch = rnn_states_batch if isinstance(rnn_states_batch, np.ndarray) else rnn_states_batch.cpu().detach().numpy()
            last_acts_batch = acts_batch

            env_acts = np.split(acts_batch, self.num_envs)
            # env step and store the relevant episode informatio
            next_obs, rewards, dones, infos = env.step(env_acts)
            if render:
                env.render()
            if training_episode:
                self.total_env_steps += self.num_envs

            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(dones_env) or t == self.episode_length - 1

            episode_obs[p_id][t] = obs
            logging.debug(f"added obs in loop at time {t}, shape {obs.shape}")
            episode_share_obs[p_id][t] = share_obs
            episode_acts[p_id][t] = np.stack(env_acts)
            episode_rewards[p_id][t] = rewards
            episode_dones[p_id][t] = dones
            episode_dones_env[p_id][t] = dones_env
            t += 1

            obs = next_obs
            logging.debug(f"next obs shape {obs.shape}")

            if terminate_episodes:
                break

        if self.epistemic_planner is not None:
            agent_poses_in_plan_at_time = self.joint_pos_in_epistemic_plan(agent_rollouts_obs_comp, t, self.num_agents)
            obs = MPERunner.add_priors_to_obs(obs, agent_poses_in_plan_at_time)
        episode_obs[p_id][t] = obs
        episode_share_obs[p_id][t] = obs.reshape(self.num_envs, -1)

        if explore:
            self.num_episodes_collected += self.num_envs
            # push all episodes collected in this rollout step to the buffer
            self.buffer.insert(self.num_envs,
                               episode_obs,
                               episode_share_obs,
                               episode_acts,
                               episode_rewards,
                               episode_dones,
                               episode_dones_env,
                               episode_avail_acts)

        average_episode_rewards = np.mean(np.sum(episode_rewards[p_id], axis=0))
        env_info['average_episode_rewards'] = average_episode_rewards

        return env_info

    # for mpe-simple_speaker_listener
    @torch.no_grad()
    def separated_collect_rollout(self, explore=True, training_episode=True, warmup=False):
        """
        Collect a rollout and store it in the buffer. Each agent has its own policy.
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        env_info = {}
        env = self.env if training_episode or warmup else self.eval_env

        obs = env.reset()

        rnn_states = np.zeros((self.num_agents, self.num_envs, self.hidden_size), dtype=np.float32)

        last_acts = {p_id : np.zeros((self.num_envs, len(self.policy_agents[p_id]), self.policies[p_id].output_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_obs = {p_id : np.zeros((self.episode_length + 1, self.num_envs, len(self.policy_agents[p_id]), self.policies[p_id].obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_share_obs = {p_id : np.zeros((self.episode_length + 1, self.num_envs, len(self.policy_agents[p_id]), self.policies[p_id].central_obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_acts = {p_id : np.zeros((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), self.policies[p_id].output_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_rewards = {p_id : np.zeros((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones = {p_id : np.ones((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones_env = {p_id : np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_avail_acts = {p_id : None for p_id in self.policy_ids}

        t = 0
        while t < self.episode_length:
            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                policy = self.policies[p_id]
                agent_obs = np.stack(obs[:, agent_id])
                share_obs = np.concatenate([obs[0, i] for i in range(self.num_agents)]).reshape(self.num_envs,
                                                                                                -1).astype(np.float32)
                # get actions for all agents to step the env
                if warmup:
                    # completely random actions in pre-training warmup phase
                    # [parallel envs, agents, dim]
                    act = policy.get_random_actions(agent_obs)
                    # get new rnn hidden state
                    _, rnn_state, _ = policy.get_actions(agent_obs,
                                                        last_acts[p_id][:, 0],
                                                        rnn_states[agent_id])
                else:
                    # get actions with exploration noise (eps-greedy/Gaussian)
                    if self.algorithm_name == "rmasac":
                        act, rnn_state, _ = policy.get_actions(agent_obs,
                                                                last_acts[p_id],
                                                                rnn_states[agent_id],
                                                                sample=explore)
                    else:
                        act, rnn_state, _ = policy.get_actions(agent_obs,
                                                                last_acts[p_id].squeeze(axis=0),
                                                                rnn_states[agent_id],
                                                                t_env=self.total_env_steps,
                                                                explore=explore)
                # update rnn hidden state
                rnn_states[agent_id] = rnn_state if isinstance(rnn_state, np.ndarray) else rnn_state.cpu().detach().numpy()
                last_acts[p_id] = np.expand_dims(act, axis=1) if isinstance(act, np.ndarray) else np.expand_dims(act.cpu().detach().numpy(), axis=1)

                episode_obs[p_id][t] = agent_obs
                episode_share_obs[p_id][t] = share_obs
                episode_acts[p_id][t] = act



            env_acts = []
            for i in range(self.num_envs):
                env_act = []
                for p_id in self.policy_ids:
                    env_act.append(last_acts[p_id][i, 0])
                env_acts.append(env_act)

            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)

            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(dones_env) or t == self.episode_length - 1
            if terminate_episodes:
                dones_env = np.ones_like(dones_env).astype(bool)


            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                episode_rewards[p_id][t] = np.expand_dims(rewards[:, agent_id], axis=1)
                episode_dones[p_id][t] = np.expand_dims(dones[:, agent_id], axis=1)
                episode_dones_env[p_id][t] = dones_env

            obs = next_obs
            t += 1

            if training_episode:
                self.total_env_steps += self.num_envs

            if terminate_episodes:
                break

        for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
            episode_obs[p_id][t] = np.stack(obs[:, agent_id])
            episode_share_obs[p_id][t] = np.concatenate([obs[0, i] for i in range(self.num_agents)]).reshape(self.num_envs,
                                                                                                -1).astype(np.float32)

        if explore:
            self.num_episodes_collected += self.num_envs
            self.buffer.insert(self.num_envs, episode_obs, episode_share_obs, episode_acts, episode_rewards, episode_dones, episode_dones_env, episode_avail_acts)

        average_episode_rewards = []
        for p_id in self.policy_ids:
            average_episode_rewards.append(np.mean(np.sum(episode_rewards[p_id], axis=0)))

        env_info['average_episode_rewards'] = np.mean(average_episode_rewards)

        return env_info

    def log(self):
        """See parent class."""
        end = time.time()
        print("\n Env {} Algo {} Exp {} runs total num timesteps {}/{}, FPS {}.\n"
              .format(self.args.scenario_name,
                      self.algorithm_name,
                      self.args.experiment_name,
                      self.total_env_steps,
                      self.num_env_steps,
                      int(self.total_env_steps / (end - self.start))))
        for p_id, train_info in zip(self.policy_ids, self.train_infos):
            self.log_train(p_id, train_info)

        self.log_env(self.env_infos)
        self.log_clear()

    def log_clear(self):
        """See parent class."""
        self.env_infos = {}

        self.env_infos['average_episode_rewards'] = []

# def main(args):
    # parser = get_config()
    # all_args = parse_args(args, parser)
    # config = {"args": all_args,
    #           "policy_info": policy_info,
    #           "policy_mapping_fn": policy_mapping_fn,
    #           "env": env,
    #           "eval_env": eval_env,
    #           "num_agents": num_agents,
    #           "device": device,
    #           "use_same_share_obs": all_args.use_same_share_obs,
    #           "run_dir": run_dir
    #           }

    # total_num_steps = 0
    # runner = MPERunner(config=config)
    # while total_num_steps < all_args.num_env_steps:
    #     total_num_steps = runner.run()

    # env.close()
    # if all_args.use_eval and (eval_env is not env):
    #     eval_env.close()

    # if all_args.use_wandb:
    #     run.finish()
    # else:
    #     runner.writter.export_scalars_to_json(
    #         str(runner.log_dir + '/summary.json'))
    #     runner.writter.close()

# if __name__ == "__main__":
#     main(sys.argv[1:])

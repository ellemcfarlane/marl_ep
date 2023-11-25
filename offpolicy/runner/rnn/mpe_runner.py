import numpy as np
import torch
import time
import sys
from offpolicy.runner.rnn.base_runner import RecRunner
from offpolicy.utils.util import setup_logging
import logging

setup_logging()

class MPERunner(RecRunner):
    """Runner class for Multiagent Particle Envs (MPE). See parent class for more information."""
    def __init__(self, config):
        super(MPERunner, self).__init__(config)
        self.collecter = self.shared_collect_rollout if self.share_policy else self.separated_collect_rollout
        # fill replay buffer with random actions
        num_warmup_episodes = max((self.batch_size, self.args.num_random_episodes))
        logging.debug("mperunner.__init__.warmup")
        # if no epistemic planner, then self is the epistemic planner
        if self.epistemic_planner is None:
            # only need to fill buffer with one episode for expert planner
            logging.info(f"collecting {1} expert episodes for qmix")
            self.collect_expert_episodes(1)
        else:
            logging.info(f"collecting {num_warmup_episodes} random episodes as warmup")
            self.collect_random_episodes(num_warmup_episodes)
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
        # only 1 policy since all agents share weights - todo (elle): why share weights if QMIX is based on IDQN? This is for centralized Q then?
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
        episode_avail_acts = {p_id : None for p_id in self.policy_ids}

        explore = False
        warmup = False
        render = True
        t = 0
        while t < self.episode_length:
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

        return env_info
    
    @staticmethod
    def get_epistemic_priors(agent_id, agent_pos, other_poses):
        """
        Get the relative positions of other agents to the given agent.
        :param agent_pos: (np.ndarray) position of the given agent.
        :param other_poses: (np.ndarray) positions of other agents.
        :return relative_pos: (np.ndarray) relative positions of other agents to the given agent.
        """
        epistemic_priors = []
        for other_agent_id, pos in enumerate(other_poses):
            if other_agent_id != agent_id:
                assert pos.shape == (2,), f"pos.shape: {pos.shape}; should be (2,)"
                assert agent_pos.shape == (2,), f"agent_pos.shape: {agent_pos.shape}; should be (2,)"
                relative_pos = pos - agent_pos
                epistemic_priors.append(relative_pos)
        return np.array(epistemic_priors)
    
    def add_priors_to_obs(self, obs, agent_poses_in_plan):
        """
        Add priors to the observation.
        :param obs: (np.ndarray) observation of shape (n_envs, n_agents, 18) to add priors to - so dim of each agent's individual obs is 18
        :param priors: (np.ndarray) priors to add to the observation.
        :return obs: (np.ndarray) observation with priors added.
        """
        env_idx = 0 # only one env for now
        # make modified_obs with same shape as obs but with priors added
        modified_obs = np.zeros((self.num_envs, self.num_agents, 18 + 2*(self.num_agents-1)))
        for agent_id in range(self.num_agents):
            # agent_pos = obs[0, agent_id, 5:7]
            # get agent_pos from state
            # env is vectorized env, so env.envs[0] is the actual env when n_envs = 1
            agent_pos = self.env.envs[env_idx].world.agents[agent_id].state.p_pos
            assert agent_poses_in_plan.shape == (self.num_agents, 2), f"agent_poses_in_plan.shape: {agent_poses_in_plan.shape}; should be {(self.num_agents, 2)}"
            priors = MPERunner.get_epistemic_priors(agent_id, agent_pos, agent_poses_in_plan)
            exp_priors_shape = (self.num_agents-1, 2)
            assert priors.shape == exp_priors_shape, f"priors.shape: {priors.shape}; should be {exp_priors_shape}"
            logging.debug(f"priors: {priors}")
            # now flattern priors to add to obs
            priors = priors.flatten()
            logging.debug(f"priors after flattening: {priors}")
            assert priors.shape == (2*(self.num_agents-1),), f"priors.shape: {priors.shape} after flattening, should be {(2*(self.num_agents-1),)}"
            old_entry = obs[env_idx, agent_id]
            logging.debug(f"old_entry.shape: {old_entry.shape}")
            logging.debug(f"priors.shape: {priors.shape}")
            new_entry = np.concatenate((old_entry, priors), axis=-1)
            assert new_entry.shape == (18 + 2*(self.num_agents-1),), f"new_entry.shape: {new_entry.shape}; should be {(18 + 2*(self.num_agents-1),)}"
            modified_obs[env_idx, agent_id] = new_entry
        return modified_obs

    def agent_poses_in_rollout_at_time(self, rollout_obss, t):
        """
        Get the agent poses in the rollout at the given time step.
        :param rollout_obss: (ndarray) of (n_agents, ep_len + 1, n_envs, obs_dim)
        :param t: (int) time step to get the agent poses at.
        :return agent_poses: (np.ndarray) agent poses at the given time step.
        """
        agent_poses = []
        env_idx = 0 # only one env for now
        for agent_id in range(self.num_agents):
            agent_obs_at_t = rollout_obss[agent_id][t][env_idx]
            assert agent_obs_at_t.shape == (18,), f"agent_obs_at_t.shape: {agent_obs_at_t.shape}; should be (18,)"
            # obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
            # shape is (2 + 2 + num_landmarks * 2 + (num_agents-1) * 2 + (num_agents-1) * 2) = 4 + 6 + 4 + 4 = 18 when num_agents = 3, num_landmarks = 3
            agent_pos = agent_obs_at_t[2:4]
            agent_poses.append(agent_pos)
        return np.array(agent_poses)

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
        # TODO (elle): what are priors at beginning?
        obs = env.reset()

        logging.debug(f'obs.shape at env.reset(): {obs.shape}')
        rnn_states_batch = np.zeros((self.num_envs * self.num_agents, self.hidden_size), dtype=np.float32)
        last_acts_batch = np.zeros((self.num_envs * self.num_agents, policy.output_dim), dtype=np.float32)

        # initialize variables to store episode information.
        episode_obs = {p_id : np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_share_obs = {p_id : np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.central_obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        if self.epistemic_planner is not None:
            epi_add_dim = 2*(self.num_agents-1)
            epi_dim = 18 + epi_add_dim
            epi_cen_dim = epi_dim*self.num_agents
            assert policy.central_obs_dim == epi_cen_dim, f"policy.central_obs_dim: {policy.central_obs_dim}; should be {epi_cen_dim}"
            assert policy.obs_dim == epi_dim, f"policy.obs_dim: {policy.obs_dim}; should be {epi_dim}"
        episode_acts = {p_id : np.zeros((self.episode_length, self.num_envs, self.num_agents, policy.output_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_rewards = {p_id : np.zeros((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones = {p_id : np.ones((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones_env = {p_id : np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_avail_acts = {p_id : None for p_id in self.policy_ids}

        logging.debug(f"episode obs shape: {episode_obs[p_id].shape}")
        t = 0
        # TODO (elle): use shared or separated collect rollout?
        # TODO is sampling from buffer random I assume? Need to get trajectory in order.
        # TODO: do we need to call this every time we call shared_collect_rollout?
        if self.epistemic_planner is not None:
            logging.debug(f"COLLECTING PRIORS")
            assert self.epistemic_planner.epistemic_planner is None
            # check agent and landmark positions of env are same as epi_env's
            epi_env = self.epistemic_planner.env.envs[0]
            init_epi_agent_poses = np.array([epi_env.world.agents[i].state.p_pos for i in range(self.num_agents)])
            init_agent_poses = np.array([env.envs[0].world.agents[i].state.p_pos for i in range(self.num_agents)])
            init_epi_landmark_poses = np.array([epi_env.world.landmarks[i].state.p_pos for i in range(self.num_agents)])
            init_landmark_poses = np.array([env.envs[0].world.landmarks[i].state.p_pos for i in range(self.num_agents)])
            if not np.all(init_epi_agent_poses == init_agent_poses):
                assert np.all(init_epi_agent_poses == init_agent_poses), f"init_epi_agent_poses: {init_epi_agent_poses} don't match init_agent_poses: {init_agent_poses}"
                assert np.all(init_epi_landmark_poses == init_landmark_poses), f"init_epi_landmark_poses: {init_epi_landmark_poses} don't match init_landmark_poses: {init_landmark_poses}"
            else:
                logging.info("nice, init conditions match!")
            # _ = self.epistemic_planner.shared_collect_rollout(explore=True, training_episode=False, warmup=False)
            n_plans = 1
            # agent rollouts is tuple of (obs, share_obs, acts, rewards, dones, dones_env, avail_acts, None, None
            # where each component is a dict mapping p_id: component
            agent_rollouts = self.epistemic_planner.buffer.sample_ordered(n_plans)
            agent_rollouts_obs_comp = agent_rollouts[0][p_id] # (3, 26, 1, 18) aka (n_agents, ep_len + 1, n_envs, obs_dim)
            # logging.debug(f"plan's obs {agent_rollouts_obs_comp.shape}, ep_len {self.episode_length}")
            # get types of each too
        while t < self.episode_length:
            if self.epistemic_planner is not None:
                # plan looks like tuple of: obs, share_obs, acts, rewards, dones, dones_env, avail_acts, None, None
                # step with planner's env and get positions of agents in planner's env to calculate priors
                logging.debug(f"BEFORE ADDING PRIORS obs: {obs.shape}, n_envs {self.num_envs}, n_agents {self.num_agents}")
                agent_poses_in_plan_at_time = self.agent_poses_in_rollout_at_time(agent_rollouts_obs_comp, t)
                # agent_poses_in_plan_at_time = np.array([[0,0], [0,0], [0,0]])
                obs = self.add_priors_to_obs(obs, agent_poses_in_plan_at_time)
                logging.debug(f"AFTER ADDING PRIORS obs: {obs.shape}, n_envs {self.num_envs}, n_agents {self.num_agents}")
            share_obs = obs.reshape(self.num_envs, -1)
            if self.epistemic_planner is not None:
                exp_shared_obs_shape = (self.num_envs, self.num_agents, 18 + 2*(self.num_agents-1))
                assert obs.shape == exp_shared_obs_shape, f"obs.shape: {obs.shape}; should be {exp_shared_obs_shape}"
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
            elif self.epistemic_planner is None: # self is the epistemic planner
                # get actions with exploration noise (eps-greedy/Gaussian)
                # TODO (elle): turn off epsilon greedy for epistemic planner collection
                acts_batch, rnn_states_batch, _ = policy.get_actions(obs_batch,
                                                                    last_acts_batch,
                                                                    rnn_states_batch,
                                                                    explore=False)
            else:
                # get actions with exploration noise (eps-greedy/Gaussian)
                # TODO (elle): turn off epsilon greedy for epistemic planner collection
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
            agent_poses_in_plan_at_time = self.agent_poses_in_rollout_at_time(agent_rollouts_obs_comp, t)
            obs = self.add_priors_to_obs(obs, agent_poses_in_plan_at_time)
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

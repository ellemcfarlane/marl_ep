import torch
import numpy as np
import torch.nn.functional as F
import copy
import itertools
from offpolicy.utils.util import huber_loss, mse_loss, to_torch
from offpolicy.utils.popart import PopArt

class MATD3:
    def __init__(self, args, num_agents, policies, policy_mapping_fn, device=None):
        self.args = args
        self.use_popart = self.args.use_popart
        self.use_value_active_masks = self.args.use_value_active_masks
        self.use_per = self.args.use_per
        self.per_eps = self.args.per_eps
        self.use_huber_loss = self.args.use_huber_loss
        self.huber_delta = self.args.huber_delta
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.num_agents = num_agents
        self.policies = policies
        self.policy_mapping_fn = policy_mapping_fn
        self.policy_ids = sorted(list(self.policies.keys()))
        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in range(self.num_agents) if self.policy_mapping_fn(agent_id) == policy_id]) for policy_id in
            self.policies.keys()}
        if self.use_popart:
            self.value_normalizer = {policy_id: PopArt(1) for policy_id in self.policies.keys()}

    def get_update_info(self, update_policy_id, obs_batch, act_batch, nobs_batch, navail_act_batch):
        cent_act = []
        cent_nact = []
        replace_ind_start = None

        # iterate through policies to get the target acts and other centralized info
        ind = 0
        for p_id in self.policy_ids:
            batch_size = obs_batch[p_id].shape[1]
            policy = self.policies[p_id]
            if p_id == update_policy_id:
                replace_ind_start = ind
            num_pol_agents = len(self.policy_agents[p_id])
            cent_act.append(list(act_batch[p_id]))

            combined_nobs_batch = np.concatenate(nobs_batch[p_id], axis=0)
            if navail_act_batch[p_id] is not None:
                combined_navail_act_batch = np.concatenate(navail_act_batch[p_id], axis=0)
            else:
                combined_navail_act_batch = None
            # use target actor to get next step actions
            with torch.no_grad():
                pol_nact, _ = policy.get_actions(combined_nobs_batch, combined_navail_act_batch, use_target=True)
                ind_agent_nacts = pol_nact.cpu().split(split_size=batch_size, dim=0)
            # cat to form the centralized next step actions
            cent_nact.append(torch.cat(ind_agent_nacts, dim=-1))

            ind += num_pol_agents

        cent_act = list(itertools.chain.from_iterable(cent_act))
        cent_nact = np.concatenate(cent_nact, axis=-1)

        return cent_act, replace_ind_start, cent_nact

    def shared_train_policy_on_batch(self, update_policy_id, batch, update_actor):
        obs_batch, cent_obs_batch, \
            act_batch, rew_batch, \
            nobs_batch, cent_nobs_batch, \
            dones_batch, dones_env_batch, \
            avail_act_batch, navail_act_batch, \
            importance_weights, idxes = batch

        train_info = {}

        cent_act, replace_ind_start, cent_nact = self.get_update_info(
            update_policy_id, obs_batch, act_batch, nobs_batch, navail_act_batch)

        cent_obs = cent_obs_batch[update_policy_id]
        cent_nobs = cent_nobs_batch[update_policy_id]
        rewards = rew_batch[update_policy_id][0]
        dones_env = dones_env_batch[update_policy_id]

        update_policy = self.policies[update_policy_id]
        batch_size = cent_obs.shape[0]

        # critic update
        with torch.no_grad():
            next_step_Q1, next_step_Q2 = update_policy.target_critic(cent_nobs, cent_nact)
            next_step_Q1, next_step_Q2 = next_step_Q1.view(-1, 1), next_step_Q2.view(-1, 1)
            next_step_Q = torch.min(next_step_Q1, next_step_Q2)

        rewards = to_torch(rewards).to(**self.tpdv).view(-1, 1)
        dones_env = to_torch(dones_env).to(**self.tpdv).view(-1, 1)

        if self.use_popart:
            target_Qs = rewards + self.args.gamma * (1 - dones_env) * self.value_normalizer[p_id].denormalize(next_step_Q)
            target_Qs = self.value_normalizer[p_id](target_Qs)
        else:
            target_Qs = rewards + self.args.gamma * (1 - dones_env) * next_step_Q

        predicted_Q1, predicted_Q2 = update_policy.critic(cent_obs, np.concatenate(cent_act, axis=-1))
        next_step_Q1, next_step_Q2 = next_step_Q1.view(-1, 1), next_step_Q2.view(-1, 1)

        error_1 = (target_Qs.detach() - predicted_Q1)
        error_2 = (target_Qs.detach() - predicted_Q2)

        if self.use_per:
            importance_weights = to_torch(importance_weights).to(**self.tpdv)
            if self.use_huber_loss:
                critic_loss_1 = huber_loss(error_1, self.huber_delta).flatten()
                critic_loss_2 = huber_loss(error_2, self.huber_delta).flatten()
            else:
                critic_loss_1 = mse_loss(error_1).flatten()
                critic_loss_2 = mse_loss(error_2).flatten()

            critic_loss_1 = (critic_loss_1 * importance_weights).mean()
            critic_loss_2 = (critic_loss_2 * importance_weights).mean()

            critic_loss = critic_loss_1 + critic_loss_2

            # new priorities are TD error
            new_priorities_1 = error_1.abs().cpu().detach().numpy().flatten()
            new_priorities_2 = error_2.abs().cpu().detach().numpy().flatten()

            new_priorities = (new_priorities_1 + new_priorities_2) / 2 + self.per_eps
        else:
            if self.use_huber_loss:
                critic_loss_1 = huber_loss(error_1, self.huber_delta).mean()
                critic_loss_2 = huber_loss(error_2, self.huber_delta).mean()
            else:
                critic_loss_1 = mse_loss(error_1).mean()
                critic_loss_2 = mse_loss(error_2).mean()

            critic_loss = critic_loss_1 + critic_loss_2
            new_priorities = None

        update_policy.critic_optimizer.zero_grad()
        critic_loss.backward()

        critic_grad_norm = torch.nn.utils.clip_grad_norm_(update_policy.critic.parameters(),
                                                                 self.args.max_grad_norm)
        update_policy.critic_optimizer.step()

        train_info['critic_loss'] = critic_loss        
        train_info['critic_grad_norm'] = critic_grad_norm

        # actor update
        if update_actor:
            for p in update_policy.critic.parameters():
                p.requires_grad = False

            num_update_agents = len(self.policy_agents[update_policy_id])
            mask_temp = []
            for p_id in self.policy_ids:
                if isinstance(self.policies[p_id].act_dim, np.ndarray):
                    # multidiscrete case
                    sum_act_dim = int(sum(self.policies[p_id].act_dim))
                else:
                    sum_act_dim = self.policies[p_id].act_dim
                for a_id in self.policy_agents[p_id]:
                    mask_temp.append(np.zeros(sum_act_dim, dtype=np.float32))

            masks = []
            done_mask = []
            # need to iterate through agents, but only formulate masks at each step
            for i in range(num_update_agents):
                curr_mask_temp = copy.deepcopy(mask_temp)
                # set the mask to 1 at locations where the action should come from the actor output
                if isinstance(update_policy.act_dim, np.ndarray):
                    # multidiscrete case
                    sum_act_dim = int(sum(update_policy.act_dim))
                else:
                    sum_act_dim = update_policy.act_dim
                curr_mask_temp[replace_ind_start + i] = np.ones(sum_act_dim, dtype=np.float32)
                curr_mask_vec = np.concatenate(curr_mask_temp)
                # expand this mask into the proper size
                curr_mask = np.tile(curr_mask_vec, (batch_size, 1))
                masks.append(curr_mask)

                # agent dones
                agent_done_batch = to_torch(dones_batch[update_policy_id][i]).to(**self.tpdv)
                done_mask.append(agent_done_batch)
            # cat to form into tensors
            mask = to_torch(np.concatenate(masks)).to(**self.tpdv)
            done_mask = torch.cat(done_mask, dim=0)
            total_batch_size = batch_size * num_update_agents
            pol_agents_obs_batch = np.concatenate(obs_batch[update_policy_id], axis=0)
            if avail_act_batch[update_policy_id] is not None:
                pol_agents_avail_act_batch = np.concatenate(avail_act_batch[update_policy_id], axis=0)
            else:
                pol_agents_avail_act_batch = None
            # get all actions from actor
            pol_acts, _ = update_policy.get_actions(pol_agents_obs_batch, pol_agents_avail_act_batch, use_gumbel=True)
            # separate into individual agent batches
            agent_actor_batches = pol_acts.split(split_size=batch_size, dim=0)

            cent_act = list(map(lambda arr: to_torch(arr).to(**self.tpdv), cent_act))
            actor_cent_acts = copy.deepcopy(cent_act)
            for i in range(num_update_agents):
                actor_cent_acts[replace_ind_start + i] = agent_actor_batches[i]

            actor_cent_acts = torch.cat(actor_cent_acts, dim=-1).repeat((num_update_agents, 1))

            # convert buffer acts to torch, formulate centralized buffer action and repeat as done above
            buffer_cent_acts = torch.cat(cent_act, dim=-1).repeat(num_update_agents, 1)

            # also repeat cent obs
            stacked_cent_obs = np.tile(cent_obs, (num_update_agents, 1))

            # combine the buffer cent acts with actor cent acts and pass into buffer
            actor_update_cent_acts = mask * actor_cent_acts + (1 - mask) * buffer_cent_acts
            actor_Qs = update_policy.critic.Q1(stacked_cent_obs, actor_update_cent_acts)
            # TODO: add mask here @Akash check this
            #actor_loss = -actor_Qs.mean()
            actor_Qs = actor_Qs * (1 - done_mask)
            actor_loss = -(actor_Qs).sum() / (1 - done_mask).sum()

            update_policy.critic_optimizer.zero_grad()
            update_policy.actor_optimizer.zero_grad()
            actor_loss.backward()

            actor_grad_norm = torch.nn.utils.clip_grad_norm_(update_policy.actor.parameters(),
                                                                    self.args.max_grad_norm)
            update_policy.actor_optimizer.step()

            for p in update_policy.critic.parameters():
                p.requires_grad = True

            train_info['actor_loss'] = actor_loss
            train_info['actor_grad_norm'] = actor_grad_norm

        return train_info, new_priorities, idxes

    def cent_train_policy_on_batch(self, update_policy_id, batch, update_actor):
        obs_batch, cent_obs_batch, \
            act_batch, rew_batch, \
            nobs_batch, cent_nobs_batch, \
            dones_batch, dones_env_batch, \
            avail_act_batch, navail_act_batch, \
            importance_weights, idxes = batch

        train_info = {}
        
        cent_act, replace_ind_start, cent_nact = self.get_update_info(
            update_policy_id, obs_batch, act_batch, nobs_batch, navail_act_batch)
        cent_obs = cent_obs_batch[update_policy_id]
        cent_nobs = cent_nobs_batch[update_policy_id]
        rewards = rew_batch[update_policy_id][0]
        dones_env = dones_env_batch[update_policy_id]
        dones = dones_batch[update_policy_id]

        update_policy = self.policies[update_policy_id]
        batch_size = obs_batch[update_policy_id].shape[1]

        num_update_agents = len(self.policy_agents[update_policy_id])

        all_agent_cent_obs = np.concatenate(cent_obs, axis=0)
        all_agent_cent_nobs = np.concatenate(cent_nobs, axis=0)
        # since this is the same for each agent, just repeat when stacking
        cent_act_buffer = np.concatenate(cent_act, axis=-1)
        all_agent_cent_act_buffer = np.tile(cent_act_buffer, (num_update_agents, 1))
        all_agent_cent_nact = np.tile(cent_nact, (num_update_agents, 1))
        all_env_dones = np.tile(dones_env, (num_update_agents, 1))
        all_agent_rewards = np.tile(rewards, (num_update_agents, 1))

        # critic update
        all_agent_rewards = to_torch(all_agent_rewards).to(**self.tpdv).reshape(-1, 1)
        all_env_dones = to_torch(all_env_dones).to(**self.tpdv).reshape(-1, 1)
        all_agent_dones = to_torch(dones).to(**self.tpdv).reshape(-1, 1)

        with torch.no_grad():
            next_step_Q1, next_step_Q2 = update_policy.target_critic(all_agent_cent_nobs, all_agent_cent_nact)
            next_step_Q1, next_step_Q2 = next_step_Q1.view(-1, 1), next_step_Q2.view(-1, 1)
            next_step_Q = torch.min(next_step_Q1, next_step_Q2)

        if self.use_popart:
            target_Qs = all_agent_rewards + self.args.gamma * \
                (1 - all_env_dones) * self.value_normalizer[p_id].denormalize(next_step_Q)
            target_Qs = self.value_normalizer[p_id](target_Qs)
        else:
            target_Qs = all_agent_rewards + self.args.gamma * (1 - all_env_dones) * next_step_Q
        predicted_Q1, predicted_Q2 = update_policy.critic(all_agent_cent_obs, all_agent_cent_act_buffer)
        predicted_Q1 = predicted_Q1.view(-1, 1)
        predicted_Q2 = predicted_Q2.view(-1, 1)

        error_1 = (target_Qs.detach() - predicted_Q1)
        error_2 = (target_Qs.detach() - predicted_Q2)

        if self.use_per:
            agent_importance_weights = np.tile(importance_weights, num_update_agents)
            agent_importance_weights = to_torch(agent_importance_weights).to(**self.tpdv)
            if self.use_huber_loss:
                critic_loss_1 = huber_loss(error_1, self.huber_delta).flatten()
                critic_loss_2 = huber_loss(error_2, self.huber_delta).flatten()
            else:
                critic_loss_1 = mse_loss(error_1).flatten()
                critic_loss_2 = mse_loss(error_2).flatten()

            critic_loss_1 = critic_loss_1 * agent_importance_weights
            critic_loss_2 = critic_loss_2 * agent_importance_weights

            if self.use_value_active_masks:
                critic_loss_1 = (critic_loss_1.view(-1, 1) * (1 - all_agent_dones)).sum() / (1 - all_agent_dones).sum()
                critic_loss_2 = (critic_loss_2.view(-1, 1) * (1 - all_agent_dones)).sum() / (1 - all_agent_dones).sum()
            else:
                critic_loss_1 = critic_loss_1.mean()
                critic_loss_2 = critic_loss_2.mean()

            critic_loss = critic_loss_1 + critic_loss_2

            # new priorities are TD error
            agent_new_priorities_1 = error_1.abs().cpu().detach().numpy().flatten()
            agent_new_priorities_2 = error_2.abs().cpu().detach().numpy().flatten()
            agent_new_priorities = (agent_new_priorities_1 + agent_new_priorities_2) / 2
            new_priorities = np.mean(np.split(agent_new_priorities, num_update_agents), axis=0) + self.per_eps
        else:
            if self.use_huber_loss:
                critic_loss_1 = huber_loss(error_1, self.huber_delta)
                critic_loss_2 = huber_loss(error_2, self.huber_delta)
            else:
                critic_loss_1 = mse_loss(error_1)
                critic_loss_2 = mse_loss(error_2)

            if self.use_value_active_masks:
                critic_loss_1 = (critic_loss_1 * (1 - all_agent_dones)).sum() / (1 - all_agent_dones).sum()
                critic_loss_2 = (critic_loss_2 * (1 - all_agent_dones)).sum() / (1 - all_agent_dones).sum()
            else:
                critic_loss_1 = critic_loss_1.mean()
                critic_loss_2 = critic_loss_2.mean()

            critic_loss = critic_loss_1 + critic_loss_2
            new_priorities = None

        update_policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(update_policy.critic.parameters(),
                                                                 self.args.max_grad_norm)
        update_policy.critic_optimizer.step()

        train_info['critic_loss'] = critic_loss        
        train_info['critic_grad_norm'] = critic_grad_norm

        # actor update
        if update_actor:
            for p in update_policy.critic.parameters():
                p.requires_grad = False

            # need to zero the critic gradient and the actor gradient since the gradients first flow through critic before getting to actor during backprop
            num_update_agents = len(self.policy_agents[update_policy_id])
            mask_temp = []
            for p_id in self.policy_ids:
                if isinstance(self.policies[p_id].act_dim, np.ndarray):
                    # multidiscrete case
                    sum_act_dim = int(sum(self.policies[p_id].act_dim))
                else:
                    sum_act_dim = self.policies[p_id].act_dim
                for a_id in self.policy_agents[p_id]:
                    mask_temp.append(np.zeros(sum_act_dim, dtype=np.float32))

            masks = []
            # TODO: FIX DONE MASK FROM HERE UNTIL LINE 166!
            done_mask = []
            # need to iterate through agents, but only formulate masks at each step
            for i in range(num_update_agents):
                curr_mask_temp = copy.deepcopy(mask_temp)
                # set the mask to 1 at locations where the action should come from the actor output
                if isinstance(update_policy.act_dim, np.ndarray):
                    # multidiscrete case
                    sum_act_dim = int(sum(update_policy.act_dim))
                else:
                    sum_act_dim = update_policy.act_dim
                curr_mask_temp[replace_ind_start + i] = np.ones(sum_act_dim, dtype=np.float32)
                curr_mask_vec = np.concatenate(curr_mask_temp)
                # expand this mask into the proper size
                curr_mask = np.tile(curr_mask_vec, (batch_size, 1))
                masks.append(curr_mask)

                # agent dones
                agent_done_batch = to_torch(dones_batch[update_policy_id][i]).to(**self.tpdv)
                done_mask.append(agent_done_batch)
            # cat to form into tensors
            mask = to_torch(np.concatenate(masks)).to(**self.tpdv)
            done_mask = torch.cat(done_mask, dim=0)

            pol_agents_obs_batch = np.concatenate(obs_batch[update_policy_id], axis=0)
            if avail_act_batch[update_policy_id] is not None:
                pol_agents_avail_act_batch = np.concatenate(avail_act_batch[update_policy_id], axis=0)
            else:
                pol_agents_avail_act_batch = None

            # get all actions from actor
            pol_acts, _ = update_policy.get_actions(pol_agents_obs_batch, pol_agents_avail_act_batch, use_gumbel=True)
            # separate into individual agent batches
            agent_actor_batches = pol_acts.split(split_size=batch_size, dim=0)

            cent_act = list(map(lambda arr: to_torch(arr).to(**self.tpdv), cent_act))
            actor_cent_acts = copy.deepcopy(cent_act)
            for i in range(num_update_agents):
                actor_cent_acts[replace_ind_start + i] = agent_actor_batches[i]

            actor_cent_acts = torch.cat(actor_cent_acts, dim=-1).repeat((num_update_agents, 1))

            # combine the buffer cent acts with actor cent acts and pass into buffer
            actor_update_cent_acts = mask * actor_cent_acts + (1 - mask) * to_torch(all_agent_cent_act_buffer).to(**self.tpdv)
            actor_Qs = update_policy.critic.Q1(all_agent_cent_obs, actor_update_cent_acts)
            # TODO: add mask
            #actor_loss = -actor_Qs.mean()
            actor_Qs = actor_Qs * (1 - done_mask)
            actor_loss = -(actor_Qs).sum() / (1 - done_mask).sum()

            update_policy.critic_optimizer.zero_grad()
            update_policy.actor_optimizer.zero_grad()
            actor_loss.backward()

            actor_grad_norm = torch.nn.utils.clip_grad_norm_(update_policy.actor.parameters(),
                                                                    self.args.max_grad_norm)
            update_policy.actor_optimizer.step()

            for p in update_policy.critic.parameters():
                p.requires_grad = True

            train_info['actor_loss'] = actor_loss
            train_info['actor_grad_norm'] = actor_grad_norm

        return train_info, new_priorities, idxes

    def prep_training(self):
        for policy in self.policies.values():
            policy.actor.train()
            policy.critic.train()
            policy.target_actor.train()
            policy.target_critic.train()

    def prep_rollout(self):
        for policy in self.policies.values():
            policy.actor.eval()
            policy.critic.eval()
            policy.target_actor.eval()
            policy.target_critic.eval()

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .model import (Actor, Critic)
from .memory import SequentialMemory
from .random_process import OrnsteinUhlenbeckProcess
from .util import *


class DDPG(nn.Module):
    def __init__(self, nb_states, nb_actions, device, num_envs, gamma, args):
        super().__init__()
        
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        
        # Create Actor and Critic Network
        self.actor = Actor(self.nb_states, self.nb_actions, args.hiddens)
        self.actor_target = Actor(self.nb_states, self.nb_actions, args.hiddens)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states, self.nb_actions, args.hiddens)
        self.critic_target = Critic(self.nb_states, self.nb_actions, args.hiddens)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize)
        self.random_process = OrnsteinUhlenbeckProcess(shape=(num_envs, nb_actions), theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = gamma
        self.depsilon = 1.0 / args.epsilon
        self.final_epsilon = args.final_epsilon

        # 
        self.epsilon = 1.0
        self.device = device

    def update_policy(self):
        # Sample batch
        np_batch = self.memory.sample_and_split(self.batch_size)

        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = list(map(lambda x: to_tensor(x, self.device), np_batch))

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target(next_state_batch, self.actor_target(next_state_batch))

        target_q_batch = reward_batch + self.discount*terminal_batch*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic(state_batch, action_batch)
        
        value_loss = F.mse_loss(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic(state_batch, self.actor(state_batch))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def observe(self, s_t, a_t, r_t, s_t1, done):
        for s, a, r, sn, d in zip(s_t, a_t, r_t, s_t1, done):
            self.memory.append(s, a, r, sn, d)

    def random_action(self, s_t):
        return np.random.uniform(-1.,1.,(s_t.shape[0], self.nb_actions))

    def greedy_action(self, s_t):
        action = to_numpy(
            self.actor(to_tensor(np.array(s_t), self.device)))
        return action

    def select_action(self, s_t):
        action = self.greedy_action(s_t)
        action += max(self.epsilon, self.final_epsilon)*self.random_process.sample()
        action = np.clip(action, -1., 1.)
        self.epsilon -= self.depsilon
        return action

    def reset(self):
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self,s):
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)

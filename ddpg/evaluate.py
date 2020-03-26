#!/usr/bin/env python3 

import numpy as np
import torch
import os

from .evaluator import Evaluator
from .ddpg import DDPG
from .util import get_output_folder
from .train import train, test

from . import parser

def run(env,
        env_normalizer,
        reload_model,
        num_eval,
        argv):
    args = parser.parse(argv)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    device = torch.device("cuda") if torch.cuda.is_available else None

    agent = DDPG(nb_states, nb_actions, device, env.num_envs, 0.99, args).to(device)
    agent.load_state_dict(reload_model['model_states'])

    evaluate = Evaluator(env, 
            env_normalizer,
            num_eval, 
            None
            )
    policy = lambda x: agent.greedy_action(x)
    evaluate(policy, save=False)


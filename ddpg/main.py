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
        eval_env,
        env_normalizer,
        gamma,
        save_dir = "",
        reload_model = None,
        argv = None):
    args = parser.parse(argv)

    env_name = eval_env.unwrapped.spec.id
    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    if save_dir != "":
        log_file = open(os.path.join(save_dir, "logs.txt"), 'w', buffering=1) 

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    device = torch.device("cuda") if torch.cuda.is_available else None

    agent = DDPG(nb_states, nb_actions, device, env.num_envs, gamma, args).to(device)
    if reload_model is not None:
        agent.load_state_dict(reload_model['model_states'])

    evaluate = Evaluator(eval_env, 
            env_normalizer,
            args.validate_episodes, 
            save_dir
            )

    train(env = env,
        env_normalizer = env_normalizer,
        agent = agent,
        train_steps = args.train_steps,
        evaluate = evaluate,
        validate_interval = args.validate_interval,
        log_interval = args.log_interval,
        save_interval = args.save_interval,
        save_dir = save_dir,
        warmup = args.warmup)

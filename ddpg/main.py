#!/usr/bin/env python3 

import numpy as np
import torch

from .evaluator import Evaluator
from .ddpg import DDPG
from .util import get_output_folder
from .train import train, test

from . import parser

def run(env, eval_env, env_normalizer, argv):
    args = parser.parse(argv)

    env_name = eval_env.unwrapped.spec.id
    args.output = get_output_folder(args.output, env_name)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(env_name)

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    device = torch.device("cuda") if torch.cuda.is_available else None

    agent = DDPG(nb_states, nb_actions, device, env.num_envs, args).to(device)
    evaluate = Evaluator(eval_env, 
            env_normalizer,
            args.validate_episodes, 
            args.validate_steps, 
            args.output
            )

    if args.mode == 'train':
        train(env, 
                env_normalizer,
                agent, 
                args.train_iter, 
                evaluate, 
                args.validate_steps, 
                args.output, 
                args.warmup,
                debug=args.debug
                )

    elif args.mode == 'test':
        test(env,
                agent,
                evaluate, 
                args.resume,
                visualize=True, 
                debug=args.debug
                )

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))

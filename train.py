import gym
import OurEnvs
import numpy as np
import time
import ddpg
import vec_envs
from env_normalize import EnvNormalizer
import a2c_ppo
import os
import torch
import sys
from arg_parser import parser

args, model_args = parser.parse_known_args()

env = vec_envs.make_vec_envs(args.env, args.env_seed, args.num_envs)
eval_env = vec_envs.make_vec_envs(args.eval_env if args.eval_env else args.env, args.env_seed, 1)

env_normalizer = EnvNormalizer(args.env_normalize_coef, env.observation_space.shape[0], args.gamma,  norm_rew=True)
reload_model = None
if args.load_model is not None:
    reload_model = torch.load(args.load_model)
    env_normalizer = reload_model['env_normalizer']

if args.save_dir != "":
    os.makedirs(args.save_dir, exist_ok = True)
    with open(os.path.join(args.save_dir, "args.txt"), "w") as arg_file:
        print(" ".join(sys.argv), file=arg_file)

if args.model == "ddpg":
    ddpg.run(env = env,
            eval_env = eval_env,
            env_normalizer = env_normalizer,
            gamma = args.gamma,
            reload_model = reload_model,
            save_dir = args.save_dir,
            argv  = model_args)
elif args.model == "ppo":
    a2c_ppo.run(env = env,
            eval_env = eval_env,
            env_normalizer = env_normalizer,
            model = "ppo",
            gamma = args.gamma,
            reload_model = reload_model,
            save_dir = args.save_dir,
            argv  = model_args)
elif args.model == "a2c":
    #a2c_ppo.run(env, eval_env, env_normalizer, "a2c", model_args)
    raise NotImplementedError



import gym
import OurEnvs
import numpy as np
import time
import argparse
import ddpg
import vec_envs
from env_normalize import EnvNormalizer
import a2c_ppo

parser = argparse.ArgumentParser()
parser.add_argument('--env', default="OurHumanoidStand-v0")
parser.add_argument('--num_envs', default=1, type=int, help="number of training environments")
parser.add_argument('--model', default="ddpg", choices=["ddpg", "ppo", "a2c"])
parser.add_argument('--env_seed', default=0, type=int)
parser.add_argument('--env_normalize_coef', default=0.9999, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
args, model_args = parser.parse_known_args()
model_args += ["--gamma", str(args.gamma)]

env = vec_envs.make_vec_envs(args.env, args.env_seed, args.num_envs)
eval_env = vec_envs.make_vec_envs(args.env, args.env_seed, 1)
env_normalizer = EnvNormalizer(args.env_normalize_coef, env.observation_space.shape[0], args.gamma,  norm_rew=True)

if args.model == "ddpg":
    ddpg.run(env, eval_env, env_normalizer, model_args)
elif args.model == "ppo":
    a2c_ppo.run(env, eval_env, env_normalizer, "ppo", model_args)
elif args.model == "a2c":
    a2c_ppo.run(env, eval_env, env_normalizer, "a2c", model_args)



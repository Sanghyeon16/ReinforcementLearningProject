import gym
import OurEnvs
import numpy as np
import time
import argparse
import ddpg
import vec_envs
from env_normalize import EnvNormalizer
import a2c_ppo
import os
import torch
import sys
from arg_parser import parser as main_parser
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder

parser = argparse.ArgumentParser()
parser.add_argument('load_model')
parser.add_argument('--env', default="OurHumanoidStand-v0")
parser.add_argument('--env-seed', default=0, type=int)
parser.add_argument('--save-dir', default=None, type=str)
parser.add_argument('--num-eval', default=10, type=int)
args = parser.parse_args()

reload_model = torch.load(args.load_model)
env_normalizer = reload_model['env_normalizer']

model_dir = os.path.dirname(args.load_model)
save_dir = args.save_dir or model_dir
os.makedirs(save_dir, exist_ok = True)

env = VecVideoRecorder(vec_envs.make_vec_envs(args.env, args.env_seed, 1), save_dir, record_video_trigger = lambda i: True, video_length=10000)

with open(os.path.join(model_dir, "args.txt"), "r") as f:
    argv = f.read().split(" ")[1:]
    _, argv = main_parser.parse_known_args(argv)

if reload_model['model'] == "ddpg":
    ddpg.evaluate(env = env,
            env_normalizer = env_normalizer,
            reload_model = reload_model,
            num_eval = args.num_eval,
            argv = argv)
elif reload_model['model'] == "ppo":
    a2c_ppo.evaluate(env = env,
            env_normalizer = env_normalizer,
            model = "ppo",
            reload_model = reload_model,
            num_eval = args.num_eval,
            argv = argv)
elif reload['model'] == "a2c":
    raise NotImplementedError

env.close()


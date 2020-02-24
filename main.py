import gym
import OurEnvs
import numpy as np
import time
import argparse
import ddpg

parser = argparse.ArgumentParser()
parser.add_argument('--env', default="OurHumanoid-v0")
parser.add_argument('--model', default="ddpg")
args, model_args = parser.parse_known_args()

env = gym.make(args.env)
eval_env = gym.make(args.env)
ddpg.run(env, eval_env, model_args)


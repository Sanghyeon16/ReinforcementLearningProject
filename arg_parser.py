import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env', default="OurHumanoidStand-v0")
parser.add_argument('--eval-env', default=None)
parser.add_argument('--num-envs', default=1, type=int, help="number of training environments")
parser.add_argument('--model', default="ddpg", choices=["ddpg", "ppo", "a2c"])
parser.add_argument('--env-seed', default=0, type=int)
parser.add_argument('--env-normalize-coef', default=0.99999, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--save-dir', default="./experiments/saved_model", type=str)
parser.add_argument('--load-model', default = None)

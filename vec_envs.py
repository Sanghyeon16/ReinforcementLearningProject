import gym
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from normalized_env import ActionNormalizedEnv

def make_env(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return ActionNormalizedEnv(env)
    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes):
    envs = [
        make_env(env_name, seed, i)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)

    return envs


import gym
import OurEnvs
import numpy as np
import time

env = gym.make("OurHumanoidThrow-v0")

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    env.render()
    print(reward)
    time.sleep(0.1)

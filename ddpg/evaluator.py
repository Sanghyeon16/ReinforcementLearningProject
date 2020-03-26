import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

from .util import *

class Evaluator(object):

    def __init__(self, env, env_normalizer, num_episodes, save_path=''):
        self.num_episodes = num_episodes
        self.save_path = save_path
        self.steps = []
        self.rews_mean = []
        self.rews_std = []
        self.env = env
        self.env_normalizer = env_normalizer

    def __call__(self, policy, debug=False, visualize=False, save=True, train_steps=0):
        observation = None
        result = []
        env = self.env
        observation = env.reset()
        assert observation is not None

        for episode in range(self.num_episodes):

            # reset at the start of episode
            episode_steps = 0
            episode_reward = 0.

            # start episode
            done = False
            while not done:
                # basic operation, action ,reward, blablabla ...
                observation = self.env_normalizer.normalize_obs(observation)
                action = policy(observation)

                observation, reward, done, info = env.step(action)
                
                if visualize:
                    env.render(mode='human')

                # update
                episode_reward += reward
                episode_steps += 1

            if debug: prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
            result.append(episode_reward)

        self.steps.append(train_steps)
        self.rews_mean.append(np.mean(result))
        self.rews_std.append(np.std(result))

        if save:
            self.save_results(os.path.join(self.save_path, "rewards.png"))
        return np.mean(result)

    def save_results(self, fn):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(self.steps, self.rews_mean, yerr=self.rews_std, fmt='-o')
        plt.savefig(fn)
        fig.clf()

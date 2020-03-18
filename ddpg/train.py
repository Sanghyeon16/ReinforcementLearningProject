#!/usr/bin/env python3 

from .util import *
from collections import deque
import numpy as np


def train(env, env_normalizer, agent, num_iterations, evaluate, validate_steps, output, warmup, visualize=True, debug=False):
    step = 0
    obs = env.reset()
    env_normalizer.update_obs(obs)
    obs = env_normalizer.normalize_obs(obs)
    episode_rewards = deque(maxlen=10)
    total_rewards = [0.0] * env.num_envs
    while step < num_iterations:
        # agent pick action ...
        if step <= warmup:
            action = agent.random_action(obs)
        else:
            action = agent.select_action(obs)
        
        # env response with next_observation, reward, terminate_info
        next_obs, reward, done, infos = env.step(action)
        for i in range(env.num_envs):
            total_rewards[i] += reward[i]
            if done[i]:
                episode_rewards.append(total_rewards[i])
                total_rewards[i] = 0

        env_normalizer.update_obs(next_obs)
        env_normalizer.update_rew(reward)
        next_obs = env_normalizer.normalize_obs(next_obs)
        reward = env_normalizer.normalize_rew(reward)

        # agent observe and update policy
        agent.observe(obs, action, reward, next_obs, done)
        if step > warmup :
            agent.update_policy()
        

        # [optional] evaluate
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
            policy = lambda x: agent.greedy_action(x)
            validate_reward = evaluate(policy, debug=False, visualize=visualize)
            if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))

        # [optional] save intermideate model
        if step % int(num_iterations/3) == 0:
            agent.save_model(output)

        # update 
        step += env.num_envs
        obs = next_obs

        if step % 1000 == 0: # end of episode
            if debug: prGreen('step:{} episode_reward:{}'.format(step, np.mean(episode_rewards)))


def test(env, agent, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.eval()
    policy = lambda x: agent.greedy_action(x)

    validate_reward = evaluate(policy, debug=debug, visualize=visualize, save=False)
    if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


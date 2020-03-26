#!/usr/bin/env python3 

from .util import *
from collections import deque
import numpy as np
import torch


def train(env,
        env_normalizer,
        agent,
        train_steps,
        evaluate,
        validate_interval,
        log_interval,
        save_interval,
        save_dir,
        warmup,
        visualize=True):
    obs = env.reset()
    env_normalizer.update_obs(obs)
    obs = env_normalizer.normalize_obs(obs)
    episode_rewards = deque(maxlen=10)
    total_rewards = [0.0] * env.num_envs
    steps = 0
    log_file = open(os.path.join(save_dir, "logs.txt"), "w", buffering=1)
    while steps < train_steps:
        # agent pick action ...
        if steps <= warmup:
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
        if steps > warmup :
            agent.update_policy()
        

        if log_interval > 0 and steps % log_interval < env.num_envs:
            msg = 'step:{} episode_reward:{}'.format(steps, np.mean(episode_rewards))
            prGreen(msg)
            print(msg, file=log_file)

        if evaluate is not None and validate_interval > 0 and steps % validate_interval < env.num_envs:
            policy = lambda x: agent.greedy_action(x)
            validate_reward = evaluate(policy, debug=False, visualize=visualize, train_steps = steps)
            msg = '[Evaluate] Step_{:07d}: mean_reward:{}'.format(steps, validate_reward)
            prYellow(msg)
            print(msg, file=log_file)

        # [optional] save intermideate model
        if save_interval > 0 and steps % save_interval < env.num_envs:
            torch.save({
                "model": "ddpg",
                "model_states": agent.state_dict(),
                "env_normalizer": env_normalizer
                },
                os.path.join(save_dir, "model-{}.pth".format(steps)))

        # update 
        steps += env.num_envs
        obs = next_obs


def test(env, agent, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.eval()
    policy = lambda x: agent.greedy_action(x)

    validate_reward = evaluate(policy, debug=debug, visualize=visualize, save=False)
    if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


#!/usr/bin/env python3 

from .util import *


def train(env, agent, num_iterations, evaluate, validate_steps, output, warmup, visualize=True, debug=False):
    step = episode = episode_steps = 0
    episode_reward = 0.
    obs = env.reset()
    agent.reset()
    while step < num_iterations:
        # agent pick action ...
        if step <= warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(obs)
        
        # env response with next_observation, reward, terminate_info
        next_obs, reward, done, info = env.step(action)

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
        step += 1
        episode_steps += 1
        episode_reward += reward
        obs = next_obs

        if done: # end of episode
            if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))

            obs = env.reset()
            agent.reset()

            # reset
            episode_steps = 0
            episode_reward = 0.
            episode += 1

def test(env, agent, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.eval()
    policy = lambda x: agent.greedy_action(x)

    validate_reward = evaluate(policy, debug=debug, visualize=visualize, save=False)
    if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


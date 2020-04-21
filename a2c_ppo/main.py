import copy
import glob
import os
import time
from collections import deque
import pickle
import matplotlib.pyplot as plt

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .a2c_ppo_acktr import algo, utils
from .a2c_ppo_acktr.algo import gail
from .a2c_ppo_acktr.arguments import get_args
from .a2c_ppo_acktr.model import Policy
from .a2c_ppo_acktr.storage import RolloutStorage
from .evaluation import evaluate


def run(env,
        eval_env,
        env_normalizer,
        model,
        gamma,
        reload_model = None,
        save_dir = "",
        argv = None):
    env_name = eval_env.unwrapped.spec.id
    args = get_args(argv)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if save_dir != "":
        log_file = open(os.path.join(save_dir, "logs.txt"), 'w', buffering=1) 

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    actor_critic = Policy(
        env.observation_space.shape,
        env.action_space,
        base_kwargs={'recurrent': args.recurrent_policy,
            'hidden_sizes': args.model_hiddens,
            'activation_fn': args.model_activation,
            'use_orth_init': args.model_orth_init})
    actor_critic.to(device)
    optimizer = optim.Adam(actor_critic.parameters(), lr=args.lr, eps=args.eps)
    if reload_model is not None:
        actor_critic.load_state_dict(reload_model['model_states'])
        optimizer.load_state_dict(reload_model['optim_states'])

    if model == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif model == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            optimizer=optimizer)
    elif model == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(env.observation_space.shape) == 1
        discr = gail.Discriminator(
            env.observation_space.shape[0] + env.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, env.num_envs,
                              env.observation_space.shape, env.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = env.reset()
    env_normalizer.update_obs(obs)
    obs = env_normalizer.normalize_obs(obs)
    obs = torch.as_tensor(obs, dtype = torch.float32, device = device)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    total_rewards = [0.0] * env.num_envs

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // env.num_envs
    total_num_steps = 0

    eval_rewards = {"steps":[], "rew_mean":[], "rew_std":[]}
    for j in range(num_updates+1):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                optimizer, j, num_updates,
                optimizer.lr if model == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = env.step(action.detach().cpu().numpy())

            for i in range(env.num_envs):
                total_rewards[i] += reward[i]
                if done[i]:
                    episode_rewards.append(total_rewards[i])
                    total_rewards[i] = 0

            env_normalizer.update_obs(obs)
            env_normalizer.update_rew(reward)
            obs = env_normalizer.normalize_obs(obs)
            reward = env_normalizer.normalize_rew(reward)

            obs = torch.as_tensor(obs, dtype = torch.float32, device = device)
            reward = torch.as_tensor(reward, dtype = torch.float32, device = device)
            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

            # save for every interval-th steps or for the last epoch
            if total_num_steps % args.save_interval < env.num_envs and save_dir != "":
                torch.save({
                    "model": model, 
                    "model_states": actor_critic.state_dict(),
                    "optim_states": optimizer.state_dict(),
                    "env_normalizer": env_normalizer
                    },
                    os.path.join(save_dir, "model-{}.pth".format(total_num_steps)))
            total_num_steps += env.num_envs

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                env.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(env)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            msg = (
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            print(msg)
            if log_file is not None:
                print(msg, file=log_file)

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            eval_rews = evaluate(actor_critic, eval_env, env_normalizer, device, log_file, visualize = not args.no_visualize)
            eval_rewards["steps"].append(total_num_steps)
            eval_rewards["rew_mean"].append(np.mean(eval_rews))
            eval_rewards["rew_std"].append(np.std(eval_rews))
            plt.errorbar(eval_rewards["steps"],
                    eval_rewards["rew_mean"],
                    yerr = eval_rewards["rew_std"]
                    )
            plt.xlabel('Steps')
            plt.ylabel('Rewards')
            plt.savefig(os.path.join(save_dir, "rewards.png"))
            plt.clf()




if __name__ == "__main__":
    main()

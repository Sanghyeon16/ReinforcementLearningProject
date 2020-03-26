import torch

from .a2c_ppo_acktr import algo, utils
from .a2c_ppo_acktr.algo import gail
from .a2c_ppo_acktr.arguments import get_args
from .a2c_ppo_acktr.model import Policy
from .a2c_ppo_acktr.storage import RolloutStorage
from .evaluation import evaluate

def run(env,
        env_normalizer,
        model,
        reload_model,
        save_dir = "",
        num_eval = 1,
        argv = None):
    args = get_args(argv)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0" if args.cuda else "cpu")

    actor_critic = Policy(
        env.observation_space.shape,
        env.action_space,
        base_kwargs={'recurrent': args.recurrent_policy,
            'hidden_sizes': args.model_hiddens,
            'activation_fn': args.model_activation,
            'use_orth_init': args.model_orth_init})
    actor_critic.to(device)
    actor_critic.load_state_dict(reload_model['model_states'])

    evaluate(actor_critic,
            env,
            env_normalizer,
            device,
            num_eval = num_eval,
            visualize=False)



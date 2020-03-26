import argparse

def parse(argv=None):
    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--hiddens', default=[400, 300], nargs="+", type=int, help='hidden num of fully connect layers')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=10000, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window-length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou-theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou-sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--ou-mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate-episodes', default=10, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--validate-interval', default=10000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--train-steps', default=10e6, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--final-epsilon', default=0.1, type=int, help='final value of epsilon for exploration')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--save-interval', type=int, default=500000, help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--log-interval', type=int, default=1000, help='log interval, one log per n updates (default: 10)')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    args = parser.parse_args(argv)
    return args

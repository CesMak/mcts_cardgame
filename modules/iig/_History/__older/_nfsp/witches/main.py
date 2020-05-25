import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import gym_witches_multiv2
import time, os
from tensorboardX import SummaryWriter

from common.utils import create_log_dir, print_args, set_global_seeds
from arguments import get_args
from train import train
from test import test

# python main.py --env 'LaserTag-small4-v0'
# python main.py --env 'Witches_multi-v2'
# python main.py --negative --dueling


def main():
    args = get_args()
    print_args(args)

    log_dir = create_log_dir(args)
    print("Log dir is:", log_dir)
    if not args.evaluate:
        writer = SummaryWriter(log_dir)

    env = gym.make(args.env)

    set_global_seeds(args.seed)
    env.seed(args.seed)

    if args.evaluate:
        test(env, args)
        env.close()
        return

    train(env, args, writer)

    writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
    writer.close()
    env.close()


if __name__ == "__main__":
    main()

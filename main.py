import argparse
import os
from core.Model import Model
from agt1.Agent import Agent as Agt1
from agt2.Agent import Agent as Agt2
import sys
from absl import flags
import numpy as np


def parse_arg():

    parser = argparse.ArgumentParser()

    # experiment settings
    parser.add_argument('-map', type=str, default='boyuCombat')
    parser.add_argument('-updates', type=int, default=int(1e3))
    parser.add_argument('-seed', type=int, default=0)

    # map settings
    parser.add_argument('-max_lives', type=float, default=np.inf)
    parser.add_argument('-max_step', type=float, default=np.inf)
    parser.add_argument('-num_timesteps', type=int, default=0)
    parser.add_argument('-max_episodes', type=int, default=0)
    parser.add_argument('-max_iters', type=int, default=int(1e8))
    parser.add_argument('-eval_timesteps_per_batch', type=int, default=5*int(1e3))

    # other settings
    parser.add_argument("-sz", type=int, default=32)
    parser.add_argument("-envs", type=int, default=2)
    parser.add_argument("-render", type=int, default=1)
    parser.add_argument("-steps", type=int, default=1)
    parser.add_argument("-run_id", type=int, default=-1)
    parser.add_argument('-output_path', type=str, default='./res/')
    parser.add_argument('-save_replay', type=bool, default=False)
    parser.add_argument('-gpu', type=str, default='0')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    FLAGS = flags.FLAGS
    FLAGS(sys.argv[:1])
    args = parse_arg()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    model = Model(Agt1, Agt2, args)

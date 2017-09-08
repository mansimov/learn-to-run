#!/usr/bin/env python
import os, sys, shutil, argparse
sys.path.append(os.getcwd())

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='linear')
parser.add_argument('--million_frames', help='How many frames to train (/ 1e6). '
    'This number gets divided by 4 due to frameskip', type=int, default=40)
args = parser.parse_args()

folder_name = "/misc/vlgscratch2/FergusGroup/mansimov/a2c/"
try:
    os.mkdir(folder_name)
except:
    pass
log_dir = os.path.join(folder_name, "{}-seed{}".format(args.env, args.seed))
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.mkdir(log_dir)
os.environ["OPENAI_LOGDIR"] = log_dir

import logging, gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.a2c.a2c import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy

def train(env_id, num_frames, seed, policy, lrschedule, num_cpu):
    num_timesteps = int(num_frames / 4 * 1.1)
    # divide by 4 due to frameskip, then do a little extras so episodes end
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and
                os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
            gym.logger.setLevel(logging.WARN)
            return wrap_deepmind(env)
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    learn(policy_fn, env, seed, total_timesteps=num_timesteps, lrschedule=lrschedule)
    env.close()

def main():

    train(args.env, num_frames=1e6 * args.million_frames, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, num_cpu=16)

if __name__ == '__main__':
    main()

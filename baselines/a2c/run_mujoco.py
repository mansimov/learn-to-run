#!/usr/bin/env python
import os, sys, shutil, argparse
sys.path.append(os.getcwd())

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='Reacher-v1')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
parser.add_argument('--million_timesteps', help='How many timesteps to train (/ 1e6).', type=int, default=1)
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
from baselines.a2c.a2c_cont import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.a2c.policies import MlpPolicy

def train(env_id, num_timesteps, seed, lrschedule, num_cpu):
    # divide by 4 due to frameskip, then do a little extras so episodes end
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and
                os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    policy_fn = MlpPolicy
    learn(policy_fn, env, seed, nsteps=5, nstack=1, total_timesteps=num_timesteps,\
        gamma=0.995, vf_coef=0.5,ent_coef=0.0001,max_grad_norm=10,
        lr=7e-4,lrschedule='linear') #constant does much worse)
    env.close()

def main():

    train(args.env, num_timesteps=int(1e6 * args.million_timesteps), seed=args.seed,
        lrschedule=args.lrschedule, num_cpu=8)

if __name__ == '__main__':
    main()

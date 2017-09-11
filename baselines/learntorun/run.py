#!/usr/bin/env python
import os, sys, shutil, argparse
print (os.getcwd())
sys.path.append(os.getcwd())

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--lrschedule', help='Learning rate schedule', default='linear', choices=['constant', 'linear'])
parser.add_argument('--million_timesteps', help='How many timesteps to train (/ 1e6).', type=int, default=1)
parser.add_argument('--lr', help="Learning rate", type=float, default=7e-4)
parser.add_argument('--max_grad_norm', help="Max grad norm", type=float, default=10)
parser.add_argument('--num_cpu', help="Num cpu in parallel", type=int, default=8)
parser.add_argument('--nsteps', help="Num of steps to rollout", type=int, default=5)
args = parser.parse_args()

head_folder_name = "/misc/vlgscratch2/FergusGroup/mansimov/a2c-learntorun/"
try:
    os.mkdir(head_folder_name)
except:
    pass
log_dir = os.path.join(head_folder_name, "learntorun-lr{}-max_grad_norm{}-nsteps{}-seed{}".format(args.lr, args.max_grad_norm, args.nsteps, args.seed))

if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.mkdir(log_dir)
os.environ["OPENAI_LOGDIR"] = log_dir

import logging, gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.learntorun.a2c_cont import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.learntorun.policies import MlpPolicy
from baselines.learntorun.learntorun_env import LearnToRunEnv

def train(num_timesteps, seed, lrschedule, num_cpu):
    def make_env(rank):
        def _thunk():
            env = LearnToRunEnv()
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and
                os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    policy_fn = MlpPolicy
    learn(policy_fn, env, seed, nsteps=5, nstack=2, total_timesteps=num_timesteps,\
        gamma=0.995, vf_coef=0.5,ent_coef=0.0001,max_grad_norm=args.max_grad_norm,
        lr=args.lr,lrschedule='linear') #constant does much worse)
    env.close()

def main():

    train(num_timesteps=int(1e6 * args.million_timesteps), seed=args.seed,
        lrschedule=args.lrschedule, num_cpu=args.num_cpu)

if __name__ == '__main__':
    main()

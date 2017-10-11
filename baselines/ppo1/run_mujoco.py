#!/usr/bin/env python
import os, sys, shutil, argparse
sys.path.append(os.getcwd())

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='Reacher-v1')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--timesteps_per_batch', type=int, default=2048)
parser.add_argument('--optim_batchsize', type=int, default=64)
parser.add_argument('--million_timesteps', type=int, default=1)
parser.add_argument('--optim_epochs', type=int, default=10)
parser.add_argument('--schedule', type=str, default='linear')


args = parser.parse_args()

folder_name = os.path.join(os.environ["checkpoint_dir"], "ppo-mpi-gpu")
try:
    os.mkdir(folder_name)
except:
    pass
log_dir = os.path.join(folder_name, "{}-seed{}".format(args.env, args.seed))
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.mkdir(log_dir)
os.environ["OPENAI_LOGDIR"] = log_dir
args.save_path = log_dir

from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
import tensorflow as tf
from util import make_env

def train(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    tf.Session(config=tf_config).__enter__()
    #U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = make_env(env_id, logger, seed)

    pposgd_simple.learn(env_id, seed, env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=args.timesteps_per_batch,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=args.optim_epochs, optim_stepsize=3e-4, optim_batchsize=args.timesteps_per_batch,
            gamma=0.99, lam=0.95, schedule=args.schedule
        )

    """
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=4096,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=512,
            gamma=0.99, lam=0.95, schedule='adapt', desired_kl=0.02,
        )
    """

    env.close()

def main():
    train(args.env, num_timesteps=args.million_timesteps*1e6, seed=args.seed)

if __name__ == '__main__':
    main()

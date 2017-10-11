#!/usr/bin/env python
import os, sys, shutil, argparse
from collections import deque
sys.path.append(os.getcwd())

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='Reacher-v1')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--num_cpu', help='Number of parallel envs', type=int, default=2)
parser.add_argument('--million_timesteps', help='Million timesteps', type=int, default=1)
parser.add_argument('--schedule', type=str, default='linear')

args = parser.parse_args()

folder_name = os.path.join(os.environ["checkpoint_dir"], "ppo-mpi-vecenv")
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
import numpy as np
import tensorflow as tf

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.safe_subproc_vec_env import SafeSubprocVecEnv

from learntorun_env import LearnToRunEnv

class StateStack(gym.Wrapper):
    def __init__(self, env, k):
        """Buffer states and stack them."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=min(env.observation_space.low),\
                                high=max(env.observation_space.high), shape=(shp[0]*k))

    def _reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        for _ in range(self.k): self.frames.append(ob)
        return self._observation()

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames)

def make_env(rank):
    def _thunk():
        if args.env.lower() == "learntorun":
            env = LearnToRunEnv(difficulty=0)
        else:
            env = gym.make(args.env)
        env.seed(args.seed + rank)
        env = bench.Monitor(env, logger.get_dir() and
            os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
        gym.logger.setLevel(logging.WARN)
        return env
        #return StateStack(env, 2)
    return _thunk

def train(env_id, num_timesteps, seed):

    from baselines.ppo1 import mlp_policy, pposgd_simple_vecenv
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    tf.Session(config=tf_config).__enter__()
    #U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)

    env = SafeSubprocVecEnv([make_env(i) for i in range(args.num_cpu)])

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    pposgd_simple_vecenv.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule=args.schedule,
        )

    env.close()

def main():
    train(args.env, num_timesteps=args.million_timesteps*1e6, seed=args.seed)

if __name__ == '__main__':
    main()

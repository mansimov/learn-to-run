#!/usr/bin/env python
import os, sys, shutil, argparse
sys.path.append(os.getcwd())
from mpi4py import MPI


import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='Humanoid-v1')
parser.add_argument('--timesteps_per_batch', help='timesteps_per_batch', type=int, default=2048)
parser.add_argument('--epochs', help='epochs', type=int, default=10)
parser.add_argument('--million_timesteps', help='million_timesteps', type=int, default=100)
parser.add_argument('--optim_batchsize', help='optim_batchsize', type=int, default=64)
parser.add_argument('--schedule', help='schedule', type=str, default='linear')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)

args = parser.parse_args()

folder_name = os.path.join(os.environ["checkpoint_dir"], "ppo-mpi")
rank = MPI.COMM_WORLD.Get_rank()
if rank == 0:
    try:
        os.mkdir(folder_name)
    except:
        pass

log_dir = os.path.join(folder_name, "{}-ntpb{}-ne{}-nob{}-nsc{}-seed{}".format(\
            args.env, args.timesteps_per_batch, args.epochs, args.optim_batchsize,\
            args.schedule, args.seed))
if rank == 0:
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

def train(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    """
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    tf.Session(config=tf_config).__enter__()
    """
    U.make_session(num_cpu=1).__enter__()
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0: logger.set_level(logger.DISABLED)
    #U.make_session(num_cpu=1).__enter__()
    workerseed = seed + MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    if args.env.lower() == "learntorun":
        from learntorun_env import LearnToRunEnv
        env = LearnToRunEnv(difficulty=0)
    else:
        env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "{}.monitor.json".format(rank)))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=args.timesteps_per_batch,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=args.epochs, optim_stepsize=3e-4, optim_batchsize=args.optim_batchsize,
            gamma=0.99, lam=0.95, schedule=args.schedule,
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

    """
    # specifically for humanoid
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=512,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=15, optim_stepsize=3e-4, optim_batchsize=4096,
            gamma=0.99, lam=0.95, schedule='adapt', # add adapt
        )
    """
    env.close()

def main():
    train(args.env, num_timesteps=args.million_timesteps*1e6, seed=args.seed)

if __name__ == '__main__':
    main()

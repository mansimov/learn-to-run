import gym, logging
from baselines import bench
import os.path as osp

def make_env(env_id, logger, seed):
    if "learntorun" in env_id.lower():
        from learntorun_env import LearnToRunEnv
        env = LearnToRunEnv(difficulty=0)
    else:
        env = gym.make(env_id)
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    return env

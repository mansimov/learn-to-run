import numpy as np
import gym
from gym import spaces, error
import sys
from osim.env import RunEnv
import random
from gym.envs.registration import EnvSpec

class LearnToRunEnv(gym.Env):
    """Wrapping LearnToRunEnv in OpenAI Gym"""
    def __init__(self, visualize=False, difficulty=None):
        super(LearnToRunEnv, self).__init__()
        if difficulty == None:
            self.difficulty = random.randint(0,2)
        else:
            self.difficulty = difficulty

        self.learntorun_env = RunEnv(visualize=visualize)
        self.observation_space = self.learntorun_env.observation_space
        self.action_space = self.learntorun_env.action_space

        self._spec = EnvSpec("RunEnv-diff{}-v1".format(difficulty))

    def _step(self, action):
        obs, reward, terminal, info = self.learntorun_env.step(action)
        return np.asarray(obs), reward, terminal, info

    def _reset(self):
        obs = self.learntorun_env.reset(difficulty=self.difficulty,\
                                            seed=self.learntorun_seed)
        return np.asarray(obs)

    def _render(self, mode='human', close=False):
        #raise NotImplementedError
        return None

    def _seed(self, seed=None):
        self.learntorun_seed = seed

    def _close(self):
        self.learntorun_env.close()

if __name__ == "__main__":
    import time
    env = LearnToRunEnv()
    env.seed(1)
    timesteps = 1000
    t1 = time.time()
    obs = env.reset()
    print (type(obs))
    for i in range(timesteps):
        obs, reward, terminal, _ = env.step(env.action_space.sample())
        print (type(obs), type(reward), type(terminal))
    t2 = time.time()
    print ("{} timesteps took {} time".format(timesteps, t2 - t1))
    print ("Done")

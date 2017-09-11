import numpy as np
import gym
from gym import spaces, error

from osim.env import RunEnv
import random

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

    def _step(self, action):
        return self.learntorun_env.step(action)

    def _reset(self):
        return self.learntorun_env.reset(difficulty=self.difficulty,\
                                            seed=self.learntorun_seed)

    def _render(self, mode='human', close=False):
        #raise NotImplementedError
        return None

    def _seed(self, seed=None):
        self.learntorun_seed = seed

    def _close(self):
        self.learntorun_env.close()

if __name__ == "__main__":
    env = LearnToRunEnv()
    env.seed(1)
    obs = env.reset()
    for i in range(200):
        obs, reward, terminal, _ = env.step(env.action_space.sample())
    print ("Done")

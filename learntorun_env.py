import numpy as np
import gym
from gym import spaces, error
import sys
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
        self.action_space = spaces.Box(low=-1.,high=1.,\
                            shape=self.learntorun_env.action_space.shape)
        self._action_space = self.learntorun_env.action_space
        #self.action_space = self.learntorun_env.action_space

    def _step(self, action):
        # rescale back to original _action_space
        scaled_action = self._action_space.low + (action + 1.) * 0.5 * (self._action_space.high - self._action_space.low)
        scaled_action = np.clip(scaled_action, self._action_space.low, self._action_space.high)
        return self.learntorun_env.step(scaled_action)

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

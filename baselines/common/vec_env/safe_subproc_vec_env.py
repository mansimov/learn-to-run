import numpy as np
from multiprocessing import Process, Queue, Pipe
from baselines.common.vec_env import VecEnv
import time

def worker(pq, cq, env_fn_wrapper):
    env = env_fn_wrapper.x()
    while True:
        cmd, data = pq.get()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            cq.put((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            cq.put(ob)
        elif cmd == 'close':
            cq.close()
            pq.close()
            del env
            break
        elif cmd == 'get_spaces':
            cq.put((env.action_space, env.observation_space))
        elif cmd == 'get_spec':
            cq.put(env.spec)
        else:
            raise NotImplementedError

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class SafeSubprocVecEnv(VecEnv):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        nenvs = len(env_fns)
        self.pqs = [Queue(1) for _ in range(nenvs)]
        self.cqs = [Queue(1) for _ in range(nenvs)]
        self.ps = [Process(target=worker, args=(pq, cq, CloudpickleWrapper(env_fn)))
            for (pq, cq, env_fn) in zip(self.pqs, self.cqs, env_fns)]
        for p in self.ps:
            p.start()

        self.pqs[0].put(('get_spaces', None))
        self.action_space, self.observation_space = self.cqs[0].get()
        self.pqs[0].put(('get_spec', None))
        self.spec = self.cqs[0].get()

    def step(self, actions):
        for pq, action in zip(self.pqs, actions):
            pq.put(('step', action))
        results = [cq.get() for cq in self.cqs]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for pq in self.pqs:
            pq.put(('reset', None))
        return np.stack([cq.get() for cq in self.cqs])

    def close(self):
        for pq in self.pqs:
            pq.put(('close', None))

        while 1:
            for p in self.ps:
                p.join(timeout=5)
            waiting = False
            for p in self.ps:
                if p.is_alive():
                    waiting = True
            if not waiting:
                break

    @property
    def num_envs(self):
        assert (len(self.pqs) == len(self.cqs))
        return len(self.pqs)

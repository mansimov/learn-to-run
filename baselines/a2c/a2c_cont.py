import os.path as osp
import gym
import time
import joblib
import logging
import numpy as np
import tensorflow as tf
from baselines import logger
from gym.spaces import Box, Discrete

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from baselines.a2c.utils import discount_with_dones, EpisodeStats
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.utils import mse

class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
            ent_coef, vf_coef, max_grad_norm, lr,
            rprop_alpha, rprop_epsilon, total_timesteps, lrschedule):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nbatch = nenvs*nsteps

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)

        A = train_model.pdtype.sample_placeholder([nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        eps = 1e-6

        # Normalising obs and returns for Mujoco.
        #nadv = ADV / (train_model.ret_rms.std + eps)
        #nr = (R - train_model.ret_rms.mean) / (train_model.ret_rms.std + eps)

        nadv = ADV
        nr = R

        nlogpac = -train_model.pd.logp(A)
        pg_loss = tf.reduce_mean(nadv * nlogpac)
        #vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vnorm), nr))
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), nr))

        entropy = tf.reduce_mean(train_model.pd.entropy())
        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=rprop_alpha, epsilon=rprop_epsilon)
        _opt_op = trainer.apply_gradients(grads)
        """
        with tf.control_dependencies([_opt_op]):
            # TODO: what about polyak updating of the rms variables
            update_ops = train_model.ob_rms._updates + train_model.ret_rms._updates #+ [ema_apply_op]
            print(update_ops)
            _train = tf.group(*update_ops)
        """
        _train = _opt_op
        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        """
        avg_norm_ret = tf.reduce_mean(tf.abs(train_model.ret_rms.mean))
        avg_norm_obs = tf.reduce_mean(tf.abs(train_model.ob_rms.mean))
        """
        def train(obs, states, returns, masks, actions, values):
            advs = returns - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            #td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:cur_lr, train_model.ret_rms.x:returns, train_model.ob_rms.x:obs}
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:cur_lr}
            if states != []:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            """
            ravg_norm_ret, ravg_norm_obs, policy_loss, value_loss, policy_entropy, _ = sess.run(
                [avg_norm_ret, avg_norm_obs, pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return ravg_norm_ret, ravg_norm_obs, policy_loss, value_loss, policy_entropy
            """
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy
        def save(save_path):
            ps = sess.run(params)
            make_path(save_path)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

def get_ac_dtype(ac_space):
    if isinstance(ac_space, Discrete):
        return np.uint8
    if isinstance(ac_space, Box):
        return np.float32
    raise NotImplementedError('do not know how to handle ac_space of type %s' % str(type(ac_space)))

class Runner(object):

    def __init__(self, env, model, nsteps, nstack, gamma, ob_dtype=np.float32):
        self.env = env
        self.model = model
        ob_space = env.observation_space
        ac_space = env.action_space

        nenv = env.num_envs
        ob_space_shape_stacked = list(ob_space.shape)
        self.nc = ob_space_shape_stacked[-1]
        ob_space_shape_stacked[-1] *= nstack
        self.obs = np.zeros(shape=[nenv] + ob_space_shape_stacked, dtype=ob_dtype)
        self.stochastic = True # hardcoded for now

        obs = env.reset()
        self.update_obs = self.update_obs_stacking if nstack > 1 else self.update_obs_no_stacking
        self.update_obs(obs)

        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        self.ob_dtype = ob_dtype
        self.ac_dtype = get_ac_dtype(ac_space)

    def update_obs_no_stacking(self, obs):
        np.copyto(self.obs, obs)

    def update_obs_stacking(self, obs):
        axis = len(self.obs.shape) - 1
        self.obs = np.roll(self.obs, shift=-self.nc, axis=axis)
        np.copyto(self.obs[..., -self.nc:], obs)

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states = self.model.step(self.stochastic, self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            for i, done in enumerate(dones):
                if done:
                    self.obs[i] = 0
            self.update_obs(obs)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.ob_dtype).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=self.ac_dtype).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        raw_rewards = mb_rewards.flatten()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_obs = mb_obs.reshape([-1] + list(mb_obs.shape[2:]))
        mb_returns = mb_rewards.flatten() # because it contains returns
        mb_actions = np.reshape(mb_actions, [-1] + list(mb_actions.shape[2:]))
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, raw_rewards, mb_returns, mb_masks, mb_actions, mb_values


def learn(policy, env, seed, nsteps, nstack, total_timesteps, gamma, vf_coef, ent_coef,
          max_grad_norm, lr, lrschedule, rprop_epsilon=1e-5, rprop_alpha=0.99, log_interval=100):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes) # HACK

    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack,
                  num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, lr=lr,
                  rprop_alpha=rprop_alpha, rprop_epsilon=rprop_epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)

    episode_stats = EpisodeStats(nsteps, nenvs)

    nbatch = nenvs*nsteps
    tstart = time.time()

    for update in range(1, total_timesteps//nbatch+1):
        obs, states, raw_rewards, returns, masks, actions, values = runner.run()
        #ravg_norm_ret, ravg_norm_obs, policy_loss, value_loss, policy_entropy = model.train(obs, states, returns, masks, actions, values)
        policy_loss, value_loss, policy_entropy = model.train(obs, states, returns, masks, actions, values)

        episode_stats.feed(raw_rewards, masks)
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            """
            logger.record_tabular("avg_norm_ret", float(ravg_norm_ret))
            logger.record_tabular("avg_norm_obs", float(ravg_norm_obs))
            """
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("policy_loss", float(policy_loss))
            logger.record_tabular("value_loss", float(value_loss * vf_coef))
            logger.record_tabular("entropy_loss", float(-1 * policy_entropy * ent_coef))
            logger.record_tabular("total_loss", float(policy_loss - policy_entropy*ent_coef + value_loss * vf_coef))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("mean_episode_length", episode_stats.mean_length())
            logger.record_tabular("mean_episode_reward", episode_stats.mean_reward())
            logger.dump_tabular()
    env.close()

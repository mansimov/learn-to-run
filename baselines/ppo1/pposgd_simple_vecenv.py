from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import copy
'''
def traj_segment_generator_vecenv(pi, env, horizon, stochastic):
    nenvs = env.num_envs
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    ac = np.repeat(np.expand_dims(ac, 0), nenvs, axis=0)
    new = [True for ne in range(nenvs)] # marks if we're on first timestep of an episode
    ob = env.reset() # because it has shape num_cpu x ob_shape

    cur_ep_ret = [0 for ne in range(nenvs)] # return in current episode
    cur_ep_len = [0 for ne in range(nenvs)] # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([copy.deepcopy(ob[0]) for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([copy.deepcopy(ac[0]) for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = copy.deepcopy(ac)
        ac, vpred = pi.batch_act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []

        for tt in range(nenvs):
            i = (t + tt) % horizon
            obs[i] = np.copy(ob[tt])
            vpreds[i] = np.copy(vpred[tt])
            news[i] = np.copy(new[tt])
            acs[i] = np.copy(ac[tt])
            prevacs[i] = np.copy(prevac[tt])

        """
        if "runenv" in env.spec.id.lower():
            clipped_ac = ac.copy()
            clipped_ac[clipped_ac<max(env.action_space.low)] = max(env.action_space.low)
            clipped_ac[clipped_ac>min(env.action_space.high)] = min(env.action_space.high)
            ob, rew, new, _ = env.step(clipped_ac)
        else:
            ob, rew, new, _ = env.step(ac)
        """
        ob, rew, new, _ = env.step(ac)

        for tt in range(nenvs):
            i = (t + tt) % horizon
            rews[i] = np.copy(rew[tt])

            cur_ep_ret[tt] += rew[tt]
            cur_ep_len[tt] += 1

            if new[tt]:
                ep_rets.append(cur_ep_ret[tt])
                ep_lens.append(cur_ep_len[tt])
                cur_ep_ret[tt] = 0
                cur_ep_len[tt] = 0
                #ob = env.reset()

        t += nenvs

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

'''
def traj_segment_generator_vecenv(pi, env, horizon, stochastic):
    t = 0
    nenvs = env.num_envs
    ac = env.action_space.sample() # not used, just so we have the datatype
    ac = np.repeat(np.expand_dims(ac, 0), nenvs, axis=0)
    new = [True for ne in range(nenvs)] # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = [0 for ne in range(nenvs)] # return in current episode
    cur_ep_len = [0 for ne in range(nenvs)] # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = [np.array([ob[0] for _ in range(horizon//nenvs)]) for _ in range(nenvs)]
    rews = [np.zeros(horizon//nenvs, 'float32') for _ in range(nenvs)]
    vpreds = [np.zeros(horizon//nenvs, 'float32') for _ in range(nenvs)]
    news = [np.zeros(horizon//nenvs, 'int32') for _ in range(nenvs)]
    acs = [np.array([ac[0] for _ in range(horizon//nenvs)]) for _ in range(nenvs)]
    prevacs = copy.deepcopy(acs)

    sub_t = 0
    while True:
        prevac = copy.deepcopy(ac)
        ac, vpred = pi.batch_act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            sub_t = 0

        i = t % horizon
        for tt in range(nenvs):
            obs[tt][sub_t] = ob[tt]
            vpreds[tt][sub_t] = vpred[tt]
            news[tt][sub_t] = new[tt]
            acs[tt][sub_t] = ac[tt]
            prevacs[tt][sub_t] = prevac[tt]

        ob, rew, new, _ = env.step(ac)

        for tt in range(nenvs):
            rews[tt][sub_t] = rew[tt]

            cur_ep_ret[tt] += rew[tt]
            cur_ep_len[tt] += 1
            if new[tt]:
                ep_rets.append(cur_ep_ret[tt])
                ep_lens.append(cur_ep_len[tt])
                cur_ep_ret[tt] = 0
                cur_ep_len[tt] = 0
        t += nenvs
        sub_t += 1

def add_vtarg_and_adv(seg, gamma, lam, nenvs):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    for tt in range(nenvs):
        new = np.append(seg["new"][tt], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
        vpred = np.append(seg["vpred"][tt], seg["nextvpred"][tt])
        T = len(seg["rew"][tt])
        if "adv" not in seg.keys():
            seg["adv"] = [None]*nenvs
        seg["adv"][tt] = gaelam = np.empty(T, 'float32')
        rew = seg["rew"][tt]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1-new[t+1]
            delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        if "tdlamret" not in seg.keys():
            seg["tdlamret"] = [None]*nenvs
        seg["tdlamret"][tt] = seg["adv"][tt] + seg["vpred"][tt]


def learn(env, policy_func,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        desired_kl=None
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    nenvs = env.num_envs
    pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - U.mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = U.mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator_vecenv(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        elif schedule == 'adapt':
            cur_lrmult = 1.0
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.next()
        add_vtarg_and_adv(seg, gamma, lam, nenvs)


        # merge all of the envs
        for k in seg.keys():
            if k != "ep_rets" and k != "ep_lens" and k != "nextvpred":
                seg[k] = np.concatenate(np.asarray(seg[k]), axis=0)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                #*newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                lossandgradout = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                newlosses, g = lossandgradout[:-1], lossandgradout[-1]
                if desired_kl != None and schedule == 'adapt':
                    if newlosses[-2] > desired_kl * 2:
                        optim_stepsize = max(1e-8, optim_stepsize / 1.5)
                    elif newlosses[-2] < desired_kl / 2:
                        optim_stepsize = min(1e0, optim_stepsize * 1.5 )
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

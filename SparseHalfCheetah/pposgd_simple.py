from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import networkx as nx

import sklearn.neighbors as nn
from graph import *
from utils import *
import gcn.globs as glob
import pdb
import time
import sys
import os



def traj_segment_generator(pi, env, horizon, adam, vfgrad, stochastic, total_gen):


    gen_graph_this_episode=total_gen # to generate or not a graph during this episode
    stats=[] # variable to keep statistics and save them on disk
    G = nx.Graph() # Graph variable
    states= [] # History of visited states
    node_ptr=0 # Pointer used to keep track of the states' list


    i_episode=0
    print('New graph at episode {}'.format(i_episode))



    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()
    states.append(ob)

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    sigmapreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred, sigmapred = pi.act(stochastic, ob)        
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "sigmapred": sigmapreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new), 
                    "nextsigmapred": sigmapred * (1-new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        sigmapreds[i] = sigmapred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac



        ob, rew, new, _ = env.step(ac)
        rews[i] = rew
        states.append(ob)
        node_ptr+=1 # Move pointer
        G.add_edge(node_ptr-1,node_ptr) # Add transition to model


        if rew and gen_graph_this_episode and len(states) > 20:

            gen_graph_this_episode=0 # Only generate one graph per episode
            total_gen=max(0,total_gen-1) # Decrease the total amount of graph generations


            # Radius-Bsaed Nearest Neighbours search to add edges
            radius = 5. 
            states = np.array(states)
            adj = nn.radius_neighbors_graph(states,radius)
            adj = adj+nx.adjacency_matrix(G)
            aug_G = nx.from_scipy_sparse_matrix(adj) # Augmented Graph


            # Identify the sources and the sinks
            source = 0 
            sink = len(states) -1
            max_sources = 40    # Max number of sources
            max_sinks=40        # Max number of sinks
            other_sources =list(range(max_sources))
            other_sinks =list(range(len(states)-max_sinks,len(states)))

            # Create the features and labels for GCN
            features = np.eye(len(states), dtype=np.float32)
            features = sparse_to_tuple(sp.lil_matrix(features))
            labels = np.zeros((len(states)))
            labels[-max_sinks:] = 1
            labels = encode_onehot(labels)


            # Diffuse the reward signal
            diffused = get_graph(aug_G.edges(),adj,features,labels,source,sink,other_sources,other_sinks)

            #Smoothen the diffused result
            interpol = make_interpolater(min(diffused),max(diffused),0,1.)
            targets = interpol(diffused) 

            #Apply to the value function           
            for epo in range(100):
                grads = vfgrad(states,targets,1.)
                adam.update(grads, 1e-3)           
            states= list(states)




        cur_ep_ret += rew
        cur_ep_len += 1
        if new:

            gen_graph_this_episode=total_gen # Reset the gen_graph variable 
            i_episode+=1
            if i_episode % 3 ==0 and gen_graph_this_episode:
                print('New graph at episode {} Remaining graphs {}'.format(i_episode,total_gen))
                G = nx.Graph()
                states= []
                node_ptr=-1 # reset pointer

            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()

            
            states.append(ob) 
            node_ptr+=1 # to avoid making edges between a terminal state and an initial state

            
        t += 1

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



def learn(env, policy_fn, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        total_gen=50,
        ):

    stats=[]

    directory= os.getcwd() + '/res'
    if not os.path.exists(directory):
        os.makedirs(directory)    
    filename = "{}/{}_graph{}_seed{}.csv".format(directory,env.spec._env_name,total_gen,glob.gcn_args.seed)

    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
    sigmaret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return for variance

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent


    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    pol_loss = - tf.reduce_mean(pi.pd.logp(ac) * atarg)

    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    sigma_loss = tf.reduce_mean(tf.square(pi.sigmapred - sigmaret))


    if optim_batchsize != timesteps_per_actorbatch:
        total_loss = pol_surr + pol_entpen + vf_loss
    else:
        total_loss = pol_loss + pol_entpen + vf_loss 
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]        

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    vfgrad = U.function([ob, ret, lrmult], U.flatgrad(vf_loss, var_list))


    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()


    saver = tf.train.Saver(max_to_keep=200)




    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch,
                                    adam, vfgrad, stochastic=True, total_gen=total_gen)


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
        else:
            raise NotImplementedError




        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]

        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate


        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        if optim_batchsize != timesteps_per_actorbatch:
            assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                # pdb.set_trace()
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))



        ###############  BOOK KEEPING
        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"],  cur_lrmult)
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


        if iters_so_far % 10 ==0:
            saver.save(tf.get_default_session(), './halfcheetahsaves{}/epoch'.format(total_gen),global_step=iters_so_far)
        stats.append(np.mean(rewbuffer))


        np.savetxt(filename,stats,delimiter=',') 

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

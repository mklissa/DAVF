#!/usr/bin/env python3

import pdb

from baselines.common import tf_util as U
from baselines import bench, logger
from baselines.common import set_global_seeds
import gym;import my_gym;

from baselines import logger
from gym.wrappers import Monitor
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import gcn.globs as g


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def mujoco_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='SparseHalfCheetah-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', help='Number of timesteps to complete training', type=int, default=int(1e6))
    parser.add_argument('--total_gen', help='Total number of graph generations (i.e. diffusion-based approximate VF generations)', type=int, default=50)


    parser.add_argument('--epochs', help='Number of epochs to train the GCN', type=int, default=50)
    parser.add_argument('--learning_rate', help='LR of the GCN', type=float, default=0.005)
    parser.add_argument('--weight_decay',help='Weight-decay of the GCN', type=float, default=1e-2)
    parser.add_argument('--hidden1',help='Hidden layer #1 of GCN', type=int, default=64)
    parser.add_argument('--hidden2',help='Hidden layer #2 of GCN', type=int, default=46)


    return parser




def make_mujoco_env(env_id, seed):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """

    set_global_seeds(seed)    
    env = gym.make(env_id)
    env.seed(seed)
        
    return env


def train(env_id, num_timesteps, seed, total_gen):
    import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = make_mujoco_env(env_id, seed)
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='constant', 
            total_gen=total_gen,
        )
    env.close()

def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    g.gcn_args=args
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
    total_gen=args.total_gen)

if __name__ == '__main__':
    main()



import pdb

import networkx as nx
import gym
import my_gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections
import scipy.sparse as sp
import matplotlib.pyplot as plt

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
import sklearn.neighbors as nn




from utils import *
from graph import *
from estimators import *





flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_integer('gen_graph', 1, 'Do you want to generate an approximate value function?')
flags.DEFINE_integer('seed', 3, 'Random seed.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train the GCN.')
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate for GCN.')
flags.DEFINE_float('weight_decay', 1e-2, 'Weight for L2 loss for GCN.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 46, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 32, 'Number of units in hidden layer 3.')


gen_graph=FLAGS.gen_graph
seed= FLAGS.seed
np.random.seed(seed)
print('seed: {} '.format(seed))


env = gym.envs.make("SparseMountainCar-v0")
env.seed(seed)




############## Initialize the Radial Basis Function for function approximation
observation_examples = np.array([env.observation_space.sample() for _ in range(10000)])

class FeaturizeState():

    def __init__(self,observation_examples):
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        self.featurizer = sklearn.pipeline.FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components=100, random_state=seed)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=100, random_state=seed)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=100, random_state=seed)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=100, random_state=seed))
                ])
        self.featurizer.fit(self.scaler.transform(observation_examples))

        self.feature_size = self.featurizer.transform(observation_examples).shape[-1]

    def __call__(self, state):
        scaled = self.scaler.transform(state)
        featurized = self.featurizer.transform(scaled)
        return featurized

featurizer = FeaturizeState(observation_examples)
##############






def actor_critic(sess, env, 
                estimator_policy, estimator_value, 
                num_episodes, discount_factor=.99):
    global gen_graph

    directory= os.getcwd() + '/res'
    if not os.path.exists(directory):
        os.makedirs(directory)
    name = "{}mountain_graph{}_seed{}.csv".format(directory,gen_graph,seed)

    G = nx.Graph()
    totsteps = 0
    stats = []
    states= []
    done=False
    node_ptr=0
    change_lr=0
    lr=1e-3         # Learning rate for the actor and the critic

    for i_episode in range(num_episodes):


        if i_episode % 3 ==0 and gen_graph:
            print('new graph')
            G = nx.Graph()
            states= []
            node_ptr=0


        if i_episode % 5 == 0 and i_episode!=0:
            np.savetxt(name,stats,delimiter=',') 

        state = env.reset()
        states.append(state) 
        rewards = 0
        losses = 0

        for t in itertools.count():
            


            action = estimator_policy.predict([state])
            next_state, reward, done, _ = env.step(action)
            rewards += reward

            node_ptr+=1
            G.add_edge(node_ptr-1,node_ptr)
            

            # Calculate TD Target
            value_next = estimator_value.predict([next_state])
            td_target = reward + (1-done) * discount_factor * value_next
            advantage = td_target - estimator_value.predict([state])
            


            estimator_value.update([state], td_target, lr)
            loss = estimator_policy.update([state], advantage, action, lr)
            losses += loss

            state = next_state
            states.append(state) 


            if done:
                node_ptr+=1 # to avoid making edges between a terminal state and a initial state
                totsteps+=t 

                print("\rEpisode {}/{} Steps {} Total Steps {} ({}) ".format(i_episode, num_episodes, t,totsteps, rewards) )
                stats.append(totsteps)
                rewards = 0



                if reward and gen_graph:

                    gen_graph=0 #Do not generate graphs anymore

                    # Adjust the aspect ratio to match the SparseMountainCar's state space
                    aspect = (0.6 + 1.2) / (2*0.07)
                    metric = lambda p0, p1: np.sqrt((p1[0] - p0[0]) * (p1[0] - p0[0]) + (p1[1] - p0[1]) * (p1[1] - p0[1]) * aspect)
                    radius = 0.02
                    real_states = np.array(states)

                    # Proceed to radius-based nearest-neighbors search to add edges
                    adj = nn.radius_neighbors_graph(real_states,radius,metric=metric)
                    adj = adj+nx.adjacency_matrix(G)
                    G_aug = nx.from_scipy_sparse_matrix(adj)


                    # Identify the sources and the sinks
                    source = 0 
                    sink = len(real_states) - 1
                    max_sources = 40
                    max_sinks=40
                    other_sources =range(max_sources)
                    other_sinks =range(len(real_states)-max_sinks,len(real_states))

                    labels = np.zeros((len(real_states)))
                    labels[-max_sinks:] = 1
                    labels = encode_onehot(labels)


                    # One-hot encoding as features - that means we don't use extra information
                    # about the collection states/nodes in order to influence diffusion.
                    features = np.eye(len(real_states), dtype=np.float32)
                    features = sparse_to_tuple(sp.lil_matrix(features))



                    # Diffuse the reward signal
                    targets = get_graph(G_aug.edges(),adj,features,labels,source,sink,other_sources,other_sinks)


                    # Update the current value function with the values obtained with GCN
                    for epo in range(30):
                        estimator_value.update(real_states, targets,lr)

                    lr=1e-4  # Reduce the learning rate to fine-tune the critic
                    

                break
            
            








with tf.Session() as sess:
    tf.set_random_seed(seed)
    global_step = tf.Variable(0, name="global_step", trainable=False)



    policy_estimator = PolicyEstimator(featurizer,env)
    value_estimator = ValueEstimator(featurizer,env)

    sess.run(tf.global_variables_initializer())
    actor_critic(sess, env, policy_estimator, value_estimator, 1000)











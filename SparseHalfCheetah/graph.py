from __future__ import division
from __future__ import print_function

import pdb
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import networkx as nx
import os
from sklearn.cluster import SpectralClustering
from sklearn import metrics

import gcn.globs as g
from utils import *
from gcn.models import GCN


colors = [(0,0,0)] + [(cm.viridis(i)) for i in range(1,256)]
new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)
FLAGS=g.gcn_args


def get_graph(edges,adj,features,labels,source,sink,other_sources,other_sinks):
    
    
    sess = tf.Session()

    
    y_train, y_val, train_mask, val_mask = get_splits(labels, source, sink, other_sources, other_sinks)


    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN


    adj =adj.toarray()
    deg = np.diag(np.sum(adj,axis=1))
    laplacian = deg - adj



    # Define placeholders
    placeholders = {
        'adj': tf.placeholder(tf.float32, shape=(None, None)) , #unnormalized adjancy matrix
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
        'learning_rate': tf.placeholder(tf.float32)
    }




    model = model_func(placeholders,edges,laplacian, input_dim=features[2][1], logging=True,FLAGS=FLAGS)


    # We now have to initialize the variables of the GCN
    gcn_vars = []
    for var in tf.global_variables():
        if 'gcn' in var.name:
            gcn_vars.append(var)
    sess.run(tf.variables_initializer(gcn_vars))
    
        

    feed_dict = construct_feed_dict(adj, features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['learning_rate']: FLAGS.learning_rate})

    cost_val = []

    start = time.time()
    for epoch in range(FLAGS.epochs):



        # We will first proceed to bring the output values close to zero before the diffusion process
        # We do this to reduce the bias introduced by the GCN's randomly initialized weights
        if epoch >1:
            feed_dict = construct_feed_dict(adj, features, support, y_train, train_mask, placeholders)
        else:
            feed_dict = construct_feed_dict(adj, features, support, y_val, val_mask, placeholders)

        feed_dict.update({placeholders['learning_rate']: FLAGS.learning_rate})

        outs = sess.run([model.opt_op, model.loss, model.accuracy,model.learning_rate], feed_dict=feed_dict)


    print("Total time for gcn {}".format(time.time()-start))
    print("Optimization Finished!")



    outputs = sess.run([tf.nn.softmax(model.outputs)], feed_dict=feed_dict)[0]
    
    return outputs[:,1]


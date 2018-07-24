import numpy as np
import tensorflow as tf


class PolicyEstimator():

    def __init__(self, featurizer, env, learning_rate=0.01, entropy_coeff=1e-3, scope="policy_estimator"):

        with tf.variable_scope(scope):

            self.state = tf.placeholder(tf.float32, [None,featurizer.feature_size], "state")
            self.action = tf.placeholder(tf.int32, name="action")
            self.advantage = tf.placeholder(dtype=tf.float32, name="advantage")
            self.lr = tf.placeholder(dtype=tf.float32, name='learnrate')
            self.featurizer = featurizer

            # This is just linear classifier
            hid = self.state

            hid = tf.contrib.layers.fully_connected(
                inputs=hid,
                num_outputs=featurizer.feature_size,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer()
                )
            
            
            self.actions = tf.contrib.layers.fully_connected(
                inputs=hid,
                num_outputs=env.action_space.n,
                activation_fn=tf.nn.softmax,
                weights_initializer=tf.contrib.layers.xavier_initializer()
                )[0]

            entropy = - tf.reduce_sum( tf.log(self.actions)*self.actions )


            self.loss = - tf.log( tf.maximum(self.actions[self.action], 1e-7)) * self.advantage
            self.loss -= entropy_coeff * entropy

            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())
    
    def predict(self,state,sess=None):

        sess = sess or tf.get_default_session()
        state = self.featurizer(state)
        actions = sess.run(self.actions, { self.state: state })
        return np.random.choice(len(actions),p=actions)

    def update(self, state, advantage, action, lr, sess=None):

        sess = sess or tf.get_default_session()
        state = self.featurizer(state)
        feed_dict = { self.state: state, self.advantage: advantage, self.action: action, self.lr: lr  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss




class ValueEstimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self, featurizer, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [None,featurizer.feature_size], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")
            self.lr = tf.placeholder(dtype=tf.float32, name='learnrate')
            self.featurizer = featurizer

            hid= self.state

            hid = tf.contrib.layers.fully_connected(
                inputs=hid,
                num_outputs=featurizer.feature_size,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer()
                )

            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=hid,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer()
                )


            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)


            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())  
                    

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        
        state = self.featurizer(state)
        val = sess.run(self.value_estimate, { self.state: state })
        return val 

    def update(self, state, target, lr, sess=None):
        sess = sess or tf.get_default_session()

        state = self.featurizer(state)
        feed_dict = { self.state: state, self.target: target, self.lr:lr }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss



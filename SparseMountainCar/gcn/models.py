from gcn.layers import *
from gcn.metrics import *
# from gcn.utils import *
import gcn.globs as g
import pdb

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        # for kwarg in kwargs.keys():
        #     assert kwarg in allowed_kwargs, 'Invbalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        # with tf.variable_scope(self.name):
        self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss,global_step=self.global_step)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCN(Model):
    def __init__(self, placeholders, edges, laplacian, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        with tf.variable_scope(self.name):
            self.edges=edges
            self.laplacian=laplacian
            self.inputs = placeholders['features']
            self.input_dim = input_dim
            self.output_dim = placeholders['labels'].get_shape().as_list()[1]
            self.placeholders = placeholders
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = placeholders['learning_rate']


            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate )
            self.build()


    def _loss(self):
        # Weight decay loss
        for layer in self.layers:
            for var in layer.vars.values():
                if "bias" in var.name:
                    continue
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        outs = tf.nn.softmax(self.outputs)[:,1]


        # first = tf.matmul(tf.transpose(tf.expand_dims(outs,1)),self.laplacian.astype(np.float32))
        # total = tf.matmul(first,tf.expand_dims(outs,1))
        # self.midloss = tf.reduce_sum(total) / len(self.edges)

        # self.loss += 1*10**(-int(FLAGS.fig))*self.midloss
        # self.loss += 1e-0*self.midloss



        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
    

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        act = tf.nn.relu
        hid1 = FLAGS.hidden1
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=act,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act=act,
                                            dropout=True,
                                            # sparse_inputs=True,
                                            logging=self.logging))



        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                            output_dim=FLAGS.hidden3,
                                            placeholders=self.placeholders,
                                            act=act,
                                            dropout=True,
                                            # sparse_inputs=True,
                                            logging=self.logging))


        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden3,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            # act=tf.nn.softmax,
                                            dropout=True,
                                            logging=self.logging))


    def predict(self):
        return tf.nn.softmax(self.outputs)

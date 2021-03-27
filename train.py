from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from scipy import sparse
#from gcn.models import GCN, MLP
from models import GCN
import scipy.io as scio
from scipy.sparse import csr_matrix
#import pdb


#Settings

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.002, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 20, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('seed',6, 'define the seed.')


# Set random seed
seed = FLAGS.seed
np.random.seed(seed)
tf.set_random_seed(seed)

print('====seed====',seed)
print('====learning_rate=====',FLAGS.learning_rate)
print('====early_stopping=====',FLAGS.early_stopping)
print('==== use 5 hidden layers with adj ====')     



names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
objects = []
for i in range(len(names)):
   with open("data/ind.{}.{}".format(FLAGS.dataset, names[i]), 'rb') as f:
        if sys.version_info > (3, 0):
             objects.append(pkl.load(f, encoding='latin1'))
        else:
             objects.append(pkl.load(f))
x, y, tx, ty, allx, ally, graph = tuple(objects)


# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

#pdb.set_trace()

### ****** all needed data would be better preserve from here ******* ###

# Some preprocessing
# features = preprocess_features(features)
# print('=== preprocess_features ===')

input_dimen = features.shape[1]
features0 = features
features, adj0 = reorgonize_features(features,adj)

features = np.expand_dims(features,-1)



nd0 = 2
if FLAGS.model == 'gcn':
    #support = [preprocess_adj(adj0)] 
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not usedouts 
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
# Define placeholders
#pdb.set_trace()
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support_adj':tf.placeholder(tf.int32, shape = (adj.shape[0],adj.shape[1])),
    'features' :tf.placeholder(tf.float32, shape=(features.shape)),
    #'features' :tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}
#pdb.set_trace()
#print('===create model====')
# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)
#model = model_func(placeholders, input_dim=input_dimen, logging=True)
# Initialize session

sess = tf.Session()
# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)   
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)

    
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())


cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy,model.outputs], feed_dict=feed_dict)
    ## the last layer of nn output is model.outputs
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)

    cost_val.append(cost)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    
    #print("########for test part##########")
    feed_dict_test = construct_feed_dict(features, support, y_test, test_mask, placeholders)
    #self.predict = tf.nn.softmax(self.outputs)
    ytest_output = sess.run([model.outputs], feed_dict = feed_dict_test)
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
        "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    
    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
       print("aaa=========",epoch)
       print("Early stopping...")
       break
    
print("Optimization Finished!")


# #pdb.set_trace()
# print("########for test part##########")
# feed_dict_test = construct_feed_dict(features, support, y_test, test_mask, placeholders)
# #self.predict = tf.nn.softmax(self.outputs)
# ytest_output = sess.run([model.outputs], feed_dict = feed_dict_test)
# test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
# print("Test set results:", "cost=", "{:.5f}".format(test_cost),
#       "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

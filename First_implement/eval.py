import os
from sklearn.metrics import f1_score
import tensorflow as tf
import numpy as np
import pandas as pd

model_dir = os.path.join('~/ml/dl/mlp', 'model')

def load_eval_data():
    eval_data = pd.read_csv('~/ml/dl/mlp/saturn_data_eval.csv', header = None)

    eval_x = np.array(eval_data[eval_data.columns[1:3]])
    eval_y = np.array(eval_data[eval_data.columns[0]])

    return eval_x, eval_y

rng = np.random.RandomState(1234)
random_state = 42

eval_x, eval_y = load_eval_data()

x = tf.placeholder(tf.float32, [None, 2], name = 'train_data')
hidden = 20

# define variable, using He Initialization
with tf.variable_scope('layer1', reuse = True) as scope:
    w1 = tf.get_variable(name = 'wight1', shape = (2, hidden),
        initializer = tf.random_normal_initializer(0, np.sqrt(2 / 2)))
    b1 = tf.get_variable(name = 'bias1', shape = (hidden),
        initializer = tf.zeros_initializer())

with tf.variable_scope('layer2', reuse = True) as scope:
    w2 = tf.get_variable(name = 'wight2', shape = (hidden, 2),
        initializer = tf.random_normal_initializer(0, np.sqrt(2 / hidden)))
    b2 = tf.get_variable(name = 'bias2', shape = (2),
        initializer = tf.zeros_initializer())

with tf.name_scope('prop'):
    u1 = tf.matmul(x, w1) + b1
    z1 = tf.nn.dropout(tf.nn.relu(u1), 0.8, seed = random_state)
    u2 = tf.nn.dropout(tf.matmul(z1, w2) + b2, 0.8, seed = random_state)
    y = tf.nn.softmax(u2)

prediction = tf.argmax(y, 1)

# load variable from saver && evaluate
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, model_dir + '/model.ckpt')
    pred_y = sess.run(prediction, feed_dict = {x: eval_x})

print(f1_score(eval_y, pred_y, average='macro'))

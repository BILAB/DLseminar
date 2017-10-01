import os
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import pandas as pd

model_dir = os.path.join('~/ml/dl/mlp', 'model')
if os.path.exists(model_dir) is False:
    os.mkdir(model_dir)

log_dir = os.path.join('~/ml/dl/mlp', 'log')
if os.path.exists(log_dir) is False:
    os.mkdir(log_dir)

def load_data():
    train_data = pd.read_csv('~/ml/dl/mlp/saturn_data_train.csv', header = None)

    train_x = np.array(train_data[train_data.columns[1:3]]).astype('float32')
    train_y = np.array(train_data[train_data.columns[0]]).astype('int32')

    return train_x, train_y


rng = np.random.RandomState(1234)
random_state = 42

train_x, train_y = load_data()

# initialize
x = tf.placeholder(tf.float32, [None, 2], name = 'train_data')
t = tf.placeholder(tf.float32, [None, 2], name = 'label')
batch_size = 50
eta = 0.001
n_epochs = 300
n_batches = train_x.shape[0] // batch_size
hidden = 20
l2_rate = 0.05

# define variable, using He Initialization
with tf.variable_scope('layer1') as scope:
    w1 = tf.get_variable(name = 'wight1', shape = (2, hidden),
        initializer = tf.random_normal_initializer(0, np.sqrt(2 / 2)))
    b1 = tf.get_variable(name = 'bias1', shape = (hidden),
        initializer = tf.zeros_initializer())

with tf.variable_scope('layer2') as scope:
    w2 = tf.get_variable(name = 'wight2', shape = (hidden, 2),
        initializer = tf.random_normal_initializer(0, np.sqrt(2 / hidden)))
    b2 = tf.get_variable(name = 'bias2', shape = (2),
        initializer = tf.zeros_initializer())

# calculate
with tf.name_scope('prop'):
    u1 = tf.matmul(x, w1) + b1
    z1 = tf.nn.dropout(tf.nn.relu(u1), 0.8, seed = random_state)
    u2 = tf.nn.dropout(tf.matmul(z1, w2) + b2, 0.8, seed = random_state)
    y = tf.nn.softmax(u2)

# define cost function
with tf.name_scope('loss'):
    cost = -tf.reduce_mean(tf.reduce_sum(t * tf.log(tf.clip_by_value(y,
        1e-10, 1.0)), axis = 1))
    # regularize with L2 cost
    l2_sqr = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
    l2cost = tf.reduce_mean(cost + l2_rate * l2_sqr)

# calculate gradient
with tf.name_scope('train'):
    train = tf.train.AdamOptimizer(eta).minimize(l2cost)

# transform to onehot expression
train_y = np.array([[1. if train_y[i] == j else 0. for j in range(2)]
    for i in range(train_y.shape[0])]).astype('float32')

# training phase && save variable
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    file_writer = tf.summary.FileWriter(log_dir, sess.graph)
    summaries = tf.summary.scalar('loss', l2cost)
    sess.run(init)
    for epoch in range(n_epochs):
        train_x, train_y = shuffle(train_x, train_y, random_state=random_state)
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            sess.run(train, feed_dict={x: train_x[start:end],
                t: train_y[start:end]})
        print(sess.run(l2cost, feed_dict={x: train_x, t: train_y}))
        summary, loss = sess.run([summaries, l2cost],
            feed_dict={x: train_x, t: train_y})
        file_writer.add_summary(summary, epoch)
    model_path = saver.save(sess, model_dir + '/model.ckpt')

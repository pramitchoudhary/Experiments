#Reference: https://pythonprogramming.net/cnn-tensorflow-convolutional-nerual-network-machine-learning-tutorial/

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import argparse
import sys


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network_model(data, no_of_classes, keep_rate):
    # Build the weight and biases dictionary for different layers
    weights = {'w_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
    'w_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
    'w_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
    'out':tf.Variable(tf.random_normal([1024, no_of_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
    'b_conv2':tf.Variable(tf.random_normal([64])),
    'b_fc':tf.Variable(tf.random_normal([1024])),
    'out':tf.Variable(tf.random_normal([no_of_classes]))}

    x = tf.reshape(data, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['w_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['w_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['w_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']
    return output


def train_neural_network(data, x_placeholder, y_placeholder, epochs, no_of_classes, keep_rate=0.5,
                         device_type='cpu:0', batch_size=128):
    with tf.device(device_type):
        prediction = convolutional_neural_network_model(x_placeholder, no_of_classes, keep_rate)
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_placeholder) )
        optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = epochs
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(data.train.num_examples/batch_size)):
                epoch_x, epoch_y = data.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x_placeholder: epoch_x, y_placeholder: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_placeholder, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x_placeholder:data.test.images, y_placeholder:data.test.labels}))


def main(_):
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    n_classes = 10
    batch_size = 128
    keep_rate = 0.8

    x_ = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    # keep_prob = tf.placeholder(tf.float32)

    train_neural_network(data=mnist, x_placeholder=x_, y_placeholder=y_, epochs=10, no_of_classes=n_classes,
                         keep_rate=keep_rate, device_type='cpu:0', batch_size=batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep_rate', type=float,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    main(parser)
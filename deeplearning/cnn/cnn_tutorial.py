# Reference: https://pythonprogramming.net/cnn-tensorflow-convolutional-nerual-network-machine-learning-tutorial/
# Reference on Optimization: Adam --> https://arxiv.org/abs/1412.6980
# Reference for hyper-parameter tuning for Deep Architecture: https://arxiv.org/abs/1206.5533

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# Enable logging ...
# tf.logging.set_verbosity(tf.logging.INFO)
from tensorflow.examples.tutorials.mnist import input_data

import argparse
import sys
import json

logs_path = '/tmp/tensorflow_logs/cnn_example'

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network_model(data, no_of_classes, keep_rate):
    # Build the weight and biases dictionary for different layers
    weights = {
    'w_conv1':tf.Variable(tf.random_normal([5,5,1,32]), name='W'),
    'w_conv2':tf.Variable(tf.random_normal([5,5,32,64]), name='W'),
    'w_fc':tf.Variable(tf.random_normal([7*7*64,1024]), name='W'),
    'out':tf.Variable(tf.random_normal([1024, no_of_classes]))
    }

    biases = {
    'b_conv1':tf.Variable(tf.random_normal([32]), name='B'),
    'b_conv2':tf.Variable(tf.random_normal([64]), name='B'),
    'b_fc':tf.Variable(tf.random_normal([1024]), name='B'),
    'out':tf.Variable(tf.random_normal([no_of_classes]))
    }

    # Transform image into 28*28 dimension
    x = tf.reshape(data, shape=[-1, 28, 28, 1])

    # First Conv-ReLu-MaxPool Layer
    conv1 = tf.nn.relu(conv2d(x, weights['w_conv1']) + biases['b_conv1'])
    tf.summary.histogram("weights", weights['w_conv1'])
    tf.summary.histogram("biases", biases['b_conv1'])
    tf.summary.histogram("activations", conv1)
    conv1 = maxpool2d(conv1)

    # Second Conv-ReLu-MaxPool Layer
    conv2 = tf.nn.relu(conv2d(conv1, weights['w_conv2']) + biases['b_conv2'])
    tf.summary.histogram("weights", weights['w_conv2'])
    tf.summary.histogram("biases", biases['b_conv2'])
    tf.summary.histogram("activations", conv2)
    conv2 = maxpool2d(conv2)

    flatten = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.matmul(flatten, weights['w_fc']) + biases['b_fc']
    fc_relu = tf.nn.relu(fc)
    tf.summary.histogram("weights", weights['w_fc'])
    tf.summary.histogram("biases", biases['b_fc'])
    tf.summary.histogram("activations", fc)
    tf.summary.histogram("fc/relu", fc_relu)

    # Helps in controlling the complexity of the model
    fc = tf.nn.dropout(fc_relu, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']
    tf.summary.histogram("output_wts", weights['out'])
    tf.summary.histogram("output_bias", biases['out'])
    return output


def train_neural_network(data, x_placeholder, y_placeholder, epochs, no_of_classes, learning_rate, device_type,
                         keep_rate=0.5, batch_size=128):

    with tf.device(device_type):
        predictions = convolutional_neural_network_model(x_placeholder, no_of_classes, keep_rate)
        print("Instance type: {}".format(type(predictions)))
        print(predictions)

        with tf.name_scope('Loss'):
            loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y_placeholder))
            tf.summary.scalar("Loss", loss_func)

        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.1).minimize(loss_func)
            tf.summary.scalar("learning_rate", learning_rate)

        with tf.name_scope('Metric'):
            # get predictions and compute accuracy
            correct_prediction = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(y_placeholder, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
            tf.summary.scalar('accuracy', accuracy)

        # Merge the summary op
        merged_summary_op = tf.summary.merge_all()

    # number of iterations
    hm_epochs = epochs
    initializer = tf.global_variables_initializer()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    sess.run(initializer)

    # write summary logs and graphs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    for epoch in range(hm_epochs):
        epoch_loss = 0
        for _ in range(int(data.train.num_examples/batch_size)):
            epoch_x, epoch_y = data.train.next_batch(batch_size)
            train_dict = {x_placeholder: epoch_x, y_placeholder: epoch_y}
            _, c, summary = sess.run([optimizer, loss_func, merged_summary_op], feed_dict=train_dict)

            # Write log at every iteration
            summary_writer.add_summary(summary, epoch)
            epoch_loss += c

        print('Epoch', epoch, 'completed out of',hm_epochs,'loss:', epoch_loss)
        # Evaluate accuracy on the test data-set
        accuracy_metric = accuracy.eval({x_placeholder:data.test.images, y_placeholder:data.test.labels}, session=sess)
        print('Accuracy on Test set (Epoch {}):'.format(epoch), accuracy_metric)

    # Evaluate accuracy on the test data-set
    accuracy_metric = accuracy.eval({x_placeholder:data.test.images, y_placeholder:data.test.labels}, session=sess)
    print('Accuracy on Test set:', accuracy_metric)
    return predictions, accuracy_metric, sess


def predict(input_images, x_buffer, prediction_inst, session_instance):
    prediction = tf.argmax(prediction_inst, axis=1)
    #values = prediction.eval({x_buffer:input_images}, session=session_instance)
    #print(values)
    return prediction


def display_results(input_data, target_label, session_instance, prediction_inst, x_buffer_placeholder):
   predicted = predict(input_data, session_instance=session_instance, prediction_inst=prediction_inst,
                  x_buffer=x_buffer_placeholder)
   #actual = target_label
   print(type(predicted))
   print(target_label)
   #print(str(acutal))
   # for i in range(6):
   #     rand_index = np.random.choice(len(input_data), 1)
   #     print("Acutal Label:{} ; Predicted Label{}".format(actual[rand_index], predicted[rand_index]))
    # n_rows = 2
    # n_cols = 3
    #
    #
    # images = np.squeeze()
    #
    # for i in range(6):
    #     plt.subplot(n_rows, n_cols, i+1)
    #     plt.imshow(np.reshape(images[i], [28, 28]), cmap='Greys_r')
    #     plt.title('True Label: ' + str(true_labels[i]) + +' ' + 'Predicted:' + str(predictions[i]), fontsize=10)
    #     frame = plt.gca()
    #     frame.axes.get_xaxis().set_visible(False)
    #     frame.axes.get_yaxis().set_visible(False)


def main(args):
    data_dir = "/tmp/data/"
    mnist = input_data.read_data_sets(data_dir, one_hot=True)

    n_classes = 10
    batch_size = 128
    keep_rate = args.keep_rate
    epochs = args.epo`ch
    l_r = args.learning_rate
    device_type = args.device_type

    # place holder for training input
    x_input = tf.placeholder(tf.float32, [None, 784])
    y_target = tf.placeholder(tf.float32, [None, 10])

    x_test_input = mnist.test.images
    y_test_target = mnist.test.labels

    print(mnist.train.num_examples)
    summary_op1 = tf.summary.text('no_of_classes', tf.convert_to_tensor('Tag1: {}'.format(n_classes)))
    summary_op2 = tf.summary.text('keep_rate', tf.convert_to_tensor('Tag2: {}'.format(keep_rate)))
    summary_op3 = tf.summary.text('batch_size', tf.convert_to_tensor('Tag3: {}'.format(batch_size)))
    summary_op4 = tf.summary.text('learning_rate', tf.convert_to_tensor('Tag4: {}'.format(l_r)))
    summary_op5 = tf.summary.text('epochs', tf.convert_to_tensor('Tag5: {}'.format(epochs)))
    summary_op6 = tf.summary.text('no_of_examples', tf.convert_to_tensor('Tag6: {}'.format(mnist.train.num_examples)))


    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for index, summary_op in enumerate([summary_op1, summary_op2, summary_op3, summary_op4, summary_op5, summary_op6]):
        text = sess.run(summary_op)
        summary_writer.add_summary(text, index)
    summary_writer.close()

    predictions_arr, acc_metric, sess_inst = train_neural_network(data=mnist, x_placeholder=x_input,
                                                    y_placeholder=y_target, epochs=epochs, no_of_classes=n_classes,
                                keep_rate=keep_rate, device_type=device_type, batch_size=batch_size, learning_rate=l_r)

    display_results(prediction_inst=predictions_arr, input_data=x_test_input, target_label=y_test_target, session_instance=sess_inst,
                    x_buffer_placeholder=x_test_input)

    #sess_inst.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep_rate', type=float,
                        default=0.8,
                        help='keep_rate')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001,
                        help='learning rate for the optimizer')
    parser.add_argument('--epsilon', type=float,
                        default=1e-08,
                        help='constant for numerical stability')
    parser.add_argument('--epoch', type=int,
                        default=10,
                        help='number of iterations')

    parser.add_argument('--device_type', type=str,
                        default='/cpu:0',
                        help='device type:''/cpu:0'' or ''/gpu:0''')
    args = parser.parse_args()

    main(args)
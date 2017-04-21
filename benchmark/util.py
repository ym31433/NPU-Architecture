'''
Arthor:      Yu-Hsuan Tseng
Date:        02/02/2017
Description: this file contains utility functions used by any
feed-forward neural network
'''
import tensorflow as tf
import numpy as np

def fast_sigmoid(x):
    return tf.div(x, (tf.add(1, tf.abs(x))))

def generate_placeholder(num_in, num_out, batch_size, type_in, type_out):
    '''generate placeholder for inputs and golden output

    Args:
        num_in: number of input neurons
        num_out: number of output neurons
        batch_size: batch size
        type_in: type of inputs, e.g. float
        type_out: type of outputs

    Returns:
        input_pl: input placeholder
        golden_pl: output placeholder
    '''
    # type
    assert(type_in == "int" or type_in == "float")
    assert(type_out == "int" or type_out == "float")
    type_in_tf = tf.float32
    if type_in == "int":
        type_in_tf = tf.int32
    type_out_tf = tf.float32
    if type_out == "int":
        type_out_tf = tf.int32
    # placeholder
    '''
    input_pl = tf.placeholder(type_in_tf,
            shape=(batch_size, num_in))
    golden_pl = tf.placeholder(type_out_tf,
            shape=(batch_size, num_out))
    '''
    input_pl = tf.placeholder(type_in_tf,
            [None, num_in])
    golden_pl = tf.placeholder(type_out_tf,
            [None, num_out])

    return input_pl, golden_pl

def fill_feed_dict(data_set, input_pl, golden_pl, batch_size):
    '''fill the feed_dict

    Args:
        data_set: input and output dataset
        input_pl: input placeholder
        golden_pl: golden output placeholder
        batch_size: batch size

    Returns:
        feed_dict: the feed dictionary mapping from placeholders to values
    '''
    input_feed, golden_feed = data_set.next_batch(batch_size)
    feed_dict = {
        input_pl: input_feed,
        golden_pl: golden_feed
    }
    # debug
    #print input_pl.get_shape()
    #print len(input_feed)
    return feed_dict

def layer(name, input_units, num_in, num_out, activation_function):
    '''calculation within a layer

    Args:
        name: the name of this layer
        input_units: input placeholder (type: Tensor)
        num_in: number of input neurons(neurons in the previous layer)
        num_out: number of output neurons(neurons within this layer)
        activation_function: the activation_function applied on outputs,
        None if nothing needs to be done

    Returns:
        output_units: output neurons(neurons within this layer)
    '''
    with tf.name_scope(name):
        # TODO: weights and biases initialization can be changed
        weights = tf.Variable(tf.zeros([num_in, num_out]),
                name='weights')
        biases = tf.Variable(tf.zeros([num_out]),
                name='biases')
        output_units = tf.matmul(input_units, weights) + biases
        if activation_function:
            output_units = activation_function(output_units)
    return output_units

def do_eval(sess, error,
        input_pl, golden_pl,
        batch_size, data_set):
    '''evaluate and print the accuracy for the given whole dataset

    Args:
        sess: the session in which the model has been trained
        error: the error for one batch of data (from benchmark)
        input_pl: input placeholder
        golden_pl: golden output placeholder
        batch_size: batch size
        data_set: the data to be evaluated
    '''
    error_sum = 0
    data_set.reset_touched()
    # steps_per_epoch is floor(data_size/batch_size)
    # num_examples is steps_per_epoch * batch_size
    num_examples, steps_per_epoch = data_set.max_steps(batch_size)
    for x in xrange(steps_per_epoch):
        #debug
        #print "iteration: %d" %x
        feed_dict = fill_feed_dict(data_set,
                input_pl, golden_pl,
                batch_size)
        error_sess = sess.run(error, feed_dict=feed_dict)
        #debug
        #print "error = %f" %error_sess
        error_sum = error_sum + error_sess
    error_mean = float(error_sum) / float(num_examples)
    print('Number of examples: %d, Error: %.3f'
            % (num_examples, error_mean))

def save_config(sess, num_layers, sim_dir, filename):
    '''save weights and biases for simulation use

    Args:
        sess: the session in which the model has been trained
        num_layers: number of layers
        sim_dir: directory to save file
    '''
    W = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    #W_test = sess.run(W)
    #print W_test
    if num_layers == 2:
        W_save, b_save = sess.run(W)
        #print W_save
        #print b_save
        np.savetxt(sim_dir+filename, np.append(W_save, b_save), delimiter=" ")
    #TODO: complete this
    elif num_layers == 3:
        W_save_h1, b_save_h1, W_save_o, b_save_o = sess.run(W)
        np.savetxt(sim_dir+filename, np.append(W_save_h1, b_save_h1), delimiter=" ")
        with open(sim_dir+filename, 'a') as fileappend:
            np.savetxt(fileappend, np.append(W_save_o, b_save_o), delimiter=" ")
    elif num_layers == 4:
        W_save_h1, b_save_h1, W_save_h2, b_save_h2, W_save_o, b_save_o = sess.run(W)
        np.savetxt(sim_dir+filename, np.append(W_save_h1, b_save_h1), delimiter=" ")
        with open(sim_dir+filename, 'a') as fileappend:
            np.savetxt(fileappend, np.append(W_save_h2, b_save_h2), delimiter=" ")
            np.savetxt(fileappend, np.append(W_save_o, b_save_o), delimiter=" ")


def save_output(sess, inputs, outputs, data_dir):
    '''save trained output

    Args:
        sess: the session in which the model has been trained
        inputs: input data fed into NN
        outputs: output
        data_dir: directory to save file
    '''
    output_save = sess.run(outputs, feed_dict={input_pl: inputs})
    np.savetxt(data_dir+"train_result.txt", output_save, delimiter=",")

# benchamrk-dependent loss function
def loss(outputs, goldens, benchmark):
    '''calculates the loss from outputs and goldens

    Args:
        outputs: [batch_size, num_out] generated by NN
        goldens: [batch_size, num_out] golden outputs
        benchmark: which benchmark is being trained

    Returns:
        loss: loss tensor of type float
    '''
    if benchmark == "hotspot" or benchmark == "hotspot_5":
        return tf.reduce_mean(tf.abs(tf.sub(outputs, goldens)))
    elif benchmark == "fft":
        return tf.reduce_mean(tf.abs(tf.sub(outputs, goldens)))
    elif benchmark == "inversek2j":
        return tf.reduce_mean(tf.abs(tf.sub(outputs, goldens)))
    else:
        return tf.reduce_mean(tf.abs(tf.sub(outputs, goldens)))

def training(loss, learning_rate):
    '''sets up the traing ops
    creates a summarizer to track the loss over time in TensorBoard
    creates an optimizer

    Args:
        loss: loss tensor, from loss()
        learning_rate: learning rate

    Return:
        train_op: the op that must be passed to the 'sess.run()'
        to cause the model to train
    '''
    # summarizer, not necessary
    tf.summary.scalar('loss', loss)
    # TODO: optimizer can be modified
    optimizer = tf.train.AdagradOptimizer(learning_rate)
    # op
    train_op = optimizer.minimize(loss)
    return train_op

# benchmark-dependent error function
def error(outputs, goldens, benchmark):
    '''accumulate the error within one batch of data

    Args:
        outputs: [batch_size, num_out] generated by NN
        goldens: [batch_size, num_out] golden outputs
        benchmark: which benchmark is being trained

    Returns:
        error: the sum of error in one batch of data
    '''
    if benchmark == "hotspot" or benchmark == "hotspot_5":
        return tf.reduce_sum(tf.abs(tf.div(tf.sub(outputs, goldens), goldens)))
    elif benchmark == "fft":
        #result = tf.reduce_sum(tf.abs(tf.sub(outputs, goldens)))
        #print result
        #return result
        return tf.reduce_sum(tf.abs(tf.sub(outputs, goldens)))
    elif benchmark == "inversek2j":
        return tf.reduce_sum(tf.abs(tf.sub(outputs, goldens)))
    else:
        return tf.reduce_sum(tf.abs(tf.div(tf.sub(outputs, goldens), goldens)))

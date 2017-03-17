import tensorflow as tf
from random import shuffle
from numpy import savetxt
import numpy as np

# setup the parameters
# TODO: these can be modified
BATCH_SIZE = 100
LEARNING_RATE = 0.02
TRAINING_FRACTION = 0.7
VALIDATING_FRACTION = 0.2

# topology
NUM_IN = 10
NUM_OUT = 1
NUM_HIDDEN1 = 4
NUM_HIDDEN2 = 0

# import input data, output golden data
inFile = open('/home/cosine/spring2017/cs533/project/hotspot/data/train.data/input.txt')
num_in_pixels = int(inFile.readline())
input_image = [ [float(i) for i in inputs.split(', ')] for inputs in inFile.readlines()]
assert(num_in_pixels == len(input_image))

outFile = open('/home/cosine/spring2017/cs533/project/hotspot/data/train.data/golden.txt')
num_out_pixels = int(outFile.readline())
output_image = [ [float(i) for i in outputs.split()] for outputs in outFile.readlines() ]
assert(num_out_pixels == len(output_image))

# prepare training data and testing data
# first random shuffle the data
assert(num_in_pixels == num_out_pixels)
indices_shuffle = range(num_in_pixels)
shuffle(indices_shuffle)
input_shuffle = [ input_image[i] for i in indices_shuffle]
output_shuffle = [ output_image[i] for i in indices_shuffle]
# then split the data into training set and testing set
train_size = int( float(num_in_pixels) * TRAINING_FRACTION )
validate_size = int( float(num_in_pixels) * VALIDATING_FRACTION )
# debug
print "total size    = %d" %num_in_pixels
print "train size    = %d" %train_size
print "validate size = %d" %validate_size
print "evaluate size = %d" %(num_in_pixels-train_size-validate_size)

# inputs
# each input has size 9
x = tf.placeholder(tf.float32, [None, NUM_IN])

# weights and biases
# TODO: initial values can be modified
W_0 = tf.Variable(tf.zeros([NUM_IN, NUM_HIDDEN1]))
b_0 = tf.Variable(tf.zeros(NUM_HIDDEN1))
W_1 = tf.Variable(tf.zeros([NUM_HIDDEN1, NUM_OUT]))
b_1 = tf.Variable(tf.zeros(NUM_OUT))

# trained outputs
#h = tf.sigmoid(tf.add(tf.matmul(x, W_0), b_0))
h = tf.add(tf.matmul(x, W_0), b_0)
y = tf.add(tf.matmul(h, W_1), b_1)

# golden outputs
y_ = tf.placeholder(tf.float32, [None, NUM_OUT])

# cost and accuracy
cost = tf.reduce_mean(tf.abs(tf.sub(y, y_)))
accuracy = 100-100*tf.reduce_mean(tf.div(tf.abs(tf.sub(y, y_)), y_))

# train
# TODO: train steps can be modified
# eg. momentum
trainstep = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(cost)

# init
init = tf.initialize_all_variables()

# session
sess = tf.Session()
sess.run(init)

# training and validating
acc = np.zeros(3)
for i in xrange(train_size/BATCH_SIZE):
    train_offset = i*BATCH_SIZE
    train_data = {
        x:input_shuffle[train_offset:train_offset+BATCH_SIZE],
        y_: output_shuffle[train_offset:train_offset+BATCH_SIZE]
    }
    sess.run(trainstep, feed_dict=train_data)
    # validation TODO: how many validation data size?
    if not i%20:
        validate_offset = train_size+i/20
        validate_data = {
            x: input_shuffle[validate_offset:validate_offset+1],
            y_: output_shuffle[validate_offset:validate_offset+1]
        }
        acc[0] = acc[1]
        acc[1] = acc[2]
        acc[2] = sess.run(accuracy, feed_dict=validate_data)
        print "validation: accuracy = %f" %acc[2]
        if acc[0] > 98 and acc[1] > 98 and acc[2] > 98: # TODO: condition modifiable
            break

# evaluation
# this evaluation is within the same iteration
evaluate_offset = train_size+validate_size
evaluate_data = {
    x: input_shuffle[evaluate_offset:-1],
    y_: output_shuffle[evaluate_offset:-1]
}
'''
# this evaluation is in different iteration
evaluate_inFile = open('/home/cosine/research/rnn_accelerator/heat_maps/simulation_output_temperatures/256x256_random_iterations0000to0031/source_256x256_random_iteration0019.csv')
evaluate_num_in_pixels = int(evaluate_inFile.readline())
evaluate_input_image = [ [float(i) for i in inputs.split(', ')] for inputs in evaluate_inFile.readlines()]
assert(evaluate_num_in_pixels == len(evaluate_input_image))

evaluate_outFile = open('/home/cosine/research/rnn_accelerator/heat_maps/simulation_output_temperatures/256x256_random_iterations0000to0031/destination_256x256_random_iteration0019.csv')
evaluate_num_out_pixels = int(evaluate_outFile.readline())
evaluate_output_image = [ [float(i) for i in outputs.split()] for outputs in evaluate_outFile.readlines() ]
assert(evaluate_num_out_pixels == len(evaluate_output_image))

evaluate_data = {
    x: evaluate_input_image[0:100],
    y_: evaluate_output_image[0:100]
}
'''
print "evaluation: accuracy = %f" %sess.run(accuracy,feed_dict=evaluate_data)

# save config, the weights and biases
'''
config_file = open("one_hidden/config.txt", 'w')
config_file.write( str(NUM_IN) + "\n") # number of input neurons
config_file.write( str(NUM_OUT) + "\n") # number of output neurons
config_file.write( str(NUM_HIDDEN1) + "\n") # number of hidden1 neurons
config_file.write( str(NUM_HIDDEN2) + "\n") # number of hidden2 neurons
config_file.close()
'''
original_data = {
    x: input_image,
}
W_0_save, W_1_save, b_0_save, b_1_save, y_save = sess.run([W_0, W_1, b_0, b_1, y], feed_dict=original_data)

W_0_save = np.concatenate((W_0_save, b_0_save[np.newaxis]), axis=0)
W_1_save = np.concatenate((W_1_save, b_1_save[np.newaxis]), axis=0)
W_0_save = np.reshape(W_0_save.T, [-1, 1])
W_1_save = np.reshape(W_1_save.T, [-1, 1])
np.savetxt("nn_config/one_hidden/weights.txt", np.append(W_0_save, W_1_save), delimiter=",")
np.savetxt("data/train.data/train_result/one_hidden.txt", y_save, delimiter=",")
'''
W_0_reshape = tf.reshape(tf.transpose(W_0), [-1, 1])
W_1_reshape = tf.reshape(tf.transpose(W_1), [-1, 1])
#debug
print W_0_reshape
print W_1_reshape
W_0_save, W_1_save, b_0_save, b_1_save = sess.run([W_0_reshape, W_1_reshape, b_0, b_1])
np.savetxt("one_hidden/weights.txt", np.append(W_0_save, W_1_save), delimiter=",")
np.savetxt("one_hidden/biases.txt", np.append(b_0_save, b_1_save), delimiter=",")
'''
'''
# debug: print the outputs
for i in range(100):
    nn_answer = sess.run(y, feed_dict={x:input_image[i:i+1]})
    print "correct answer:"
    print output_image[i]
    print "nn answer:"
    print nn_answer
'''

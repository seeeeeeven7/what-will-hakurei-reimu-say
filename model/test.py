import os
import argparse
import os
import random
import time
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Network:

    def __init__(self, in_size, lstm_size, num_layers, out_size, session,
                 learning_rate=0.003, name="rnn"):
        self.scope = name
        self.in_size = in_size
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.out_size = out_size
        self.session = session
        self.learning_rate = tf.constant(learning_rate)
        # Last state of LSTM, used when running the network in TEST mode
        self.lstm_last_state = np.zeros(
            (self.num_layers * 2 * self.lstm_size,)
        )
        with tf.variable_scope(self.scope):
            # (batch_size, timesteps, in_size)
            self.xinput = tf.placeholder(
                tf.float32,
                shape=(None, None, self.in_size),
                name="xinput"
            )
            self.lstm_init_value = tf.placeholder(
                tf.float32,
                shape=(None, self.num_layers * 2 * self.lstm_size),
                name="lstm_init_value"
            )
            # LSTM
            self.lstm_cells = [
                tf.contrib.rnn.BasicLSTMCell(
                    self.lstm_size,
                    forget_bias=1.0,
                    state_is_tuple = True
                ) for i in range(self.num_layers)
            ]
            self.lstm = tf.contrib.rnn.MultiRNNCell(
                self.lstm_cells,
                state_is_tuple=False
            )
            # Iteratively compute output of recurrent network
            outputs, self.lstm_new_state = tf.nn.dynamic_rnn(
                self.lstm,
                self.xinput,
                initial_state=self.lstm_init_value,
                dtype=tf.float32
            )
            # Linear activation (FC layer on top of the LSTM net)
            self.rnn_out_W = tf.Variable(
                tf.random_normal(
                    (self.lstm_size, self.out_size),
                    stddev=0.01
                )
            )
            self.rnn_out_B = tf.Variable(
                tf.random_normal(
                    (self.out_size,), stddev=0.01
                )
            )
            outputs_reshaped = tf.reshape(outputs, [-1, self.lstm_size])
            network_output = tf.matmul(
                outputs_reshaped,
                self.rnn_out_W
            ) + self.rnn_out_B
            batch_time_shape = tf.shape(outputs)
            self.final_outputs = tf.reshape(
                tf.nn.softmax(network_output),
                (batch_time_shape[0], batch_time_shape[1], self.out_size)
            )
            # Training: provide target outputs for supervised training.
            self.y_batch = tf.placeholder(
                tf.float32,
                (None, None, self.out_size)
            )
            y_batch_long = tf.reshape(self.y_batch, [-1, self.out_size])
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=network_output,
                    labels=y_batch_long
                )
            )
            self.train_op = tf.train.RMSPropOptimizer(
                self.learning_rate,
                0.9
            ).minimize(self.cost)

    def run_step(self, x, init_zero_state=True):
        # Reset the initial state of the network.
        if init_zero_state:
            init_value = np.zeros((self.num_layers * 2 * self.lstm_size,))
        else:
            init_value = self.lstm_last_state
        out, next_lstm_state = self.session.run(
            [self.final_outputs, self.lstm_new_state],
            feed_dict={
                self.xinput: [x],
                self.lstm_init_value: [init_value]
            }
        )
        self.lstm_last_state = next_lstm_state[0]
        return out[0][0]


def encode1hot(s, vocab):
    embed = np.zeros((len(s), len(vocab)))
    cnt = 0
    for index, ch in enumerate(s):
        embed[index][vocab.index(ch)] = 1.0
    return embed

def decode1hot(v, vocab):
    return vocab[v.index(1)]

def load_data(input):
    # Load the data
    lines = []
    with open(input, 'r', encoding = 'utf8') as f:
        lines = f.read().splitlines()
    # Calc vocabularies
    vocab = set()
    for line in lines:
        vocab |= set(line)
    vocab = sorted(list(vocab))
    # Convert to 1-hot
    for index, line in enumerate(lines):
        lines[index] = encode1hot(line, vocab)
    return lines, vocab

def main():

    # Files
    data_file = '../data/talk_in_game/all_withoutspace.txt'
    save_file = 'saved/model.ckpt'

    # Load the data
    lines, vocab = load_data(data_file)

    # Model parameters
    in_size = out_size = len(vocab)
    lstm_size = 64
    num_layers = 2
    batch_size = 64
    time_steps = 5
    learning_rate = 0.003

    NUM_TRAIN_BATCHES = 20000

    # Number of test characters of text to generate after training the network
    LEN_TEST_TEXT = 20

    # Initialize the network
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config = config)
    network = Network(
        in_size = in_size,
        lstm_size = lstm_size,
        num_layers = num_layers,
        out_size = out_size,
        session = sess,
        learning_rate = learning_rate,
        name = "char_rnn_network"
    )
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, save_file)

    #     TEST_PREFIX = TEST_PREFIX.lower()
    TEST_PREFIX = '巫女'
    for i in range(len(TEST_PREFIX)):
        out = network.run_step(encode1hot(TEST_PREFIX[i], vocab), i == 0)

    print("Sentence:")
    gen_str = TEST_PREFIX
    for i in range(LEN_TEST_TEXT):
        # Sample character from the network according to the generated
        # output probabilities.
        element = np.random.choice(range(len(vocab)), p=out)
        gen_str += vocab[element]
        out = network.run_step(encode1hot(vocab[element], vocab), False)

    print(gen_str)


if __name__ == "__main__":
    main()

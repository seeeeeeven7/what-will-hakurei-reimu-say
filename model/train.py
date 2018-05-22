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
                    state_is_tuple=False
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
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=network_output,
                    labels=y_batch_long
                )
            )
            self.train_op = tf.train.RMSPropOptimizer(
                self.learning_rate,
                0.9
            ).minimize(self.cost)

    # xbatch must be (batch_size, timesteps, input_size)
    # ybatch must be (batch_size, timesteps, output_size)
    def train_batch(self, xbatch, ybatch):
        init_value = np.zeros(
            (xbatch.shape[0], self.num_layers * 2 * self.lstm_size)
        )
        cost, _ = self.session.run(
            [self.cost, self.train_op],
            feed_dict={
                self.xinput: xbatch,
                self.y_batch: ybatch,
                self.lstm_init_value: init_value
            }
        )
        return cost


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


def check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('saved/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

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
    LEN_TEST_TEXT = 50

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
    check_restore_parameters(sess, saver)

    # Train the network
    last_time = time.time()
    batch_x = np.zeros((batch_size, time_steps, in_size))
    batch_y = np.zeros((batch_size, time_steps, in_size))
    possible_line_ids = []
    for index, line in enumerate(lines):
        if len(line) > time_steps:
            possible_line_ids.append(index)

    for batch_index in range(NUM_TRAIN_BATCHES):
        # Sampling
        for i in range(batch_size):
            line_id = random.choice(possible_line_ids)
            pos1 = random.randint(0, len(lines[line_id]) - time_steps - 1)
            pos2 = pos1 + 1
            for j in range(time_steps):
                batch_x[i][j][:] = lines[line_id][pos1 + j][:]
                batch_y[i][j][:] = lines[line_id][pos2 + j][:]
        # Train step
        cost = network.train_batch(batch_x, batch_y)
        if batch_index % 100 == 0:
            new_time = time.time()
            diff = new_time - last_time
            last_time = new_time
            print("batch: {}  loss: {}  speed: {} batches / s".format(batch_index, cost, 100 / diff))
            saver.save(sess, save_file)

    # # 1) TRAIN THE NETWORK
    # if args.mode == "train":
    #     check_restore_parameters(sess, saver)
    #     last_time = time.time()
    #     batch = np.zeros((batch_size, time_steps, in_size))
    #     batch_y = np.zeros((batch_size, time_steps, in_size))
    #     possible_batch_ids = range(data.shape[0] - time_steps - 1)

    #     for i in range(NUM_TRAIN_BATCHES):
    #         # Sample time_steps consecutive samples from the dataset text file
    #         batch_id = random.sample(possible_batch_ids, batch_size)

    #         for j in range(time_steps):
    #             ind1 = [k + j for k in batch_id]
    #             ind2 = [k + j + 1 for k in batch_id]

    #             batch[:, j, :] = data[ind1, :]
    #             batch_y[:, j, :] = data[ind2, :]

    #         cst = net.train_batch(batch, batch_y)

    #         if (i % 100) == 0:
    #             new_time = time.time()
    #             diff = new_time - last_time
    #             last_time = new_time
    #             print("batch: {}  loss: {}  speed: {} batches / s".format(
    #                 i, cst, 100 / diff
    #             ))
    #             saver.save(sess, ckpt_file)
    # elif args.mode == "talk":
    #     # 2) GENERATE LEN_TEST_TEXT CHARACTERS USING THE TRAINED NETWORK
    #     saver.restore(sess, ckpt_file)

    #     TEST_PREFIX = TEST_PREFIX.lower()
    #     for i in range(len(TEST_PREFIX)):
    #         out = net.run_step(embed_to_vocab(TEST_PREFIX[i], vocab), i == 0)

    #     print("Sentence:")
    #     gen_str = TEST_PREFIX
    #     for i in range(LEN_TEST_TEXT):
    #         # Sample character from the network according to the generated
    #         # output probabilities.
    #         element = np.random.choice(range(len(vocab)), p=out)
    #         gen_str += vocab[element]
    #         out = net.run_step(embed_to_vocab(vocab[element], vocab), False)

    #     print(gen_str)


if __name__ == "__main__":
    main()

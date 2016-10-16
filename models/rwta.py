import tensorflow as tf
import keras.backend as K
from keras.layers import Conv2D, Input
from utils import load, save
learning_rate = .001
beta1 = .9
obj_size = 8


def wta(X):
    M = K.max(X, axis=(1, 2), keepdims=True)
    R = K.cast(K.equal(X, M), 'float32')

    return R*X


def conv_lstm(X, H, C, WX, WH):
    Z = WX(X) + WH(H)
    It, Ft, Ot, Gt = tf.split(3, 4, Z)
    It = tf.nn.sigmoid(It)
    Ft = tf.nn.sigmoid(Ft)
    Ot = tf.nn.sigmoid(Ot)
    Gt = tf.nn.tanh(Gt)
    C = Ft * C + It * Gt
    H = Ot * tf.nn.tanh(C)
    return H, C


def get_model(sess, name, time_len=10, batch_size=32, image_size=32):
    checkpoint_dir = './outputs/results_' + name
    nb_filter = 128
    state_size = (batch_size, image_size, image_size, nb_filter)
    with tf.variable_scope(name):
        X = Input(shape=(time_len, image_size, image_size, 3))
        Y = Input(batch_shape=(batch_size, time_len, image_size, image_size, 3))
        conv0 = Conv2D(8, 3, 3, activation='relu', border_mode='same', name='l0')
        convH = Conv2D(nb_filter*4, 3, 3, activation='linear', border_mode='same', name='lh')
        convX = Conv2D(nb_filter*4, 3, 3, bias=False, activation='linear', border_mode='same', name='lx')
        conv1 = Conv2D(3, 7, 7, activation='tanh', border_mode='same', name='l1')

        def step(x, states):
            h_tm1, c_tm1 = states[:2]
            x_t = conv0(x)
            h_t, c_t = conv_lstm(x_t, h_tm1, c_tm1, convX, convH)
            w_t = wta(h_t)
            y_t = conv1(w_t)

            return y_t, [h_t, c_t]

        initial_states = [K.zeros(state_size), K.zeros(state_size)]
        last_, Out, states = K.rnn(step, X, initial_states, input_length=time_len, unroll=True)

        mvars = [L.trainable_weights for L in [conv0, convH, convX, conv1]]
        mvars = [item for sublist in mvars for item in sublist]
        print mvars

        cost = tf.reduce_mean(tf.square(Y - Out))
        optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(cost, var_list=mvars)
        tf.initialize_all_variables().run()

    w1 = conv1.W
    writer = tf.train.SummaryWriter("./outputs/logs_{}".format(name), K.get_session().graph)
    saver = tf.train.Saver()
    sum_loss = tf.scalar_summary("loss", cost)
    sum_gen = tf.image_summary("Out", tf.reshape(Out, (-1, image_size, image_size, 1)))
    sum_loc = tf.image_summary("Loc", tf.reshape(states[1], (-1, image_size, image_size, 1)))
    sum_s1 = tf.image_summary("s1", w1)
    sums = tf.merge_summary([sum_loss, sum_gen, sum_loc, sum_s1])

    def ftrain(x, y, counter, sess=sess):
        outs, loss, wsums, _ = sess.run([Out, cost, sums, optim], feed_dict={X:x, Y:y})
        writer.add_summary(wsums, counter)
        return outs, loss

    def ftest(x, y, sess=sess):
        outs, loss = sess.run([Out, cost], feed_dict={X:x, Y:y})
        return outs, loss

    def f_load():
        load(sess, saver, checkpoint_dir, name)

    def f_save(step):
        save(sess, saver, checkpoint_dir, 0, name)

    return ftrain, ftest, f_load, f_save, []


import tensorflow._api.v2.compat.v1 as tf
import tf_slim as slim
import h5py
import os
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.disable_v2_behavior()

classes = 2


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def weights_variables(shape):
    weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    return weight


def bias_variables(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_norm_layer(value, is_training):
    if is_training is True:
        return tf.keras.layers.BatchNormalization()(value, training=True)
    else:
        return tf.keras.layers.BatchNormalization()(value, training=False)


def inception_module_1st(net, scope, filters_num, filters_size):
    with tf.variable_scope(scope):
        with tf.variable_scope('bh1'):
            bh1 = slim.conv2d(net, filters_num[0], [1, filters_size[0]], scope='bh1_conv1_1x3')
            bh1 = slim.conv2d(bh1, filters_num[0], [filters_size[0], 1], scope='bh1_conv1_3x1')
        with tf.variable_scope('bh2'):
            bh2 = slim.conv2d(net, filters_num[1], [1, filters_size[1]], scope='bh2_conv2_1x5')
            bh2 = slim.conv2d(bh2, filters_num[1], [filters_size[1], 1], scope='bh2_conv2_5x1')
        with tf.variable_scope('bh3'):
            bh3 = slim.conv2d(net, filters_num[2], [1, filters_size[2]], scope='bh3_conv2_1x7')
            bh3 = slim.conv2d(bh3, filters_num[2], [filters_size[2], 1], scope='bh3_conv2_7x1')
        with tf.variable_scope('bh4'):
            bh4 = slim.conv2d(net, filters_num[3], [1, filters_size[3]], scope='bh4_conv2_1x9')
            bh4 = slim.conv2d(bh4, filters_num[3], [filters_size[3], 1], scope='bh4_conv2_9x1')
        with tf.variable_scope('bh5'):
            bh5 = slim.conv2d(net, filters_num[4], [1, filters_size[4]], scope='bh4_conv2_1x11')
            bh5 = slim.conv2d(bh5, filters_num[4], [filters_size[4], 1], scope='bh4_conv2_11x1')
        net = tf.concat([bh1, bh2, bh3, bh4, bh5], axis=3)
    return net


def inception_module_2ed(net, scope, filters_num):
    with tf.variable_scope(scope):
        with tf.variable_scope('l1'):
            # slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,rate=rate, padding='VALID', scope=scope)
            bran1 = slim.conv2d(net, filters_num[0], [1, 1], scope='l1_conv1_1x1')
        with tf.variable_scope('l2'):
            bran2 = slim.conv2d(net, filters_num[1], [1, 1], scope='l2_conv1_1x1')
            bran2 = slim.conv2d(bran2, filters_num[2], [1, 3], scope='l2_conv2_1x3')
            bran2 = slim.conv2d(bran2, filters_num[2], [3, 1], scope='l2_conv2_3x1')
        with tf.variable_scope('l3'):
            bran3 = slim.conv2d(net, filters_num[3], [1, 1], scope='l3_conv1_1x1')
            bran3 = slim.conv2d(bran3, filters_num[4], [1, 5], scope='l3_conv2_1x5')
            bran3 = slim.conv2d(bran3, filters_num[4], [5, 1], scope='l3_conv2_5x1')
        with tf.variable_scope('l4'):
            bran4 = slim.conv2d(net, filters_num[5], [1, 1], scope='l4_conv1_1x1')
            bran4 = slim.conv2d(bran4, filters_num[6], [1, 7], scope='l4_conv2_1x7')
            bran4 = slim.conv2d(bran4, filters_num[6], [7, 1], scope='l4_conv2_7x1')
        with tf.variable_scope('l5'):
            bran5 = slim.max_pool2d(net, [3, 3], scope='l5_max_3x3', padding='SAME')
            bran5 = slim.conv2d(bran5, filters_num[7], [1, 1], scope='l5_conv_1x1')
        net = tf.concat([bran1, bran2, bran3, bran4, bran5], axis=3)
    return net


# inputs:50 * 1600
def Spec_CNN(inputs, training, keep_prob):
    with tf.variable_scope('spec_cnn'):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_regularizer=slim.l2_regularizer(5e-4),
                weights_initializer=slim.xavier_initializer(),
        ):
            with slim.arg_scope(
                    [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                    padding='SAME',
                    stride=1,
            ):
                net1 = batch_norm_layer(inception_module_1st(net=inputs, filters_num=[16, 8, 4, 2, 2],
                                                             filters_size=[3, 5, 7, 9, 11],
                                                             scope='1st_inception_model'),
                                        is_training=training)
                net1 = slim.max_pool2d(net1, 3, stride=2, scope='max_polling1')
                net1 = tf.nn.dropout(net1, keep_prob)

                net2 = batch_norm_layer(inception_module_2ed(net=net1, scope='2ed_inception_model',
                                                             filters_num=[4, 16, 32, 8, 16, 4, 8, 4]),
                                        is_training=training)
                net2 = slim.max_pool2d(net2, 3, stride=2, scope='max_polling2')
                net2 = tf.nn.dropout(net2, keep_prob)

                net = tf.reshape(net2, [-1, 13 * 400 * 64])
                return net


def single_train(component_seq):
    print('-----------------------------------------The component seq : ', component_seq, '-----------------------------------------')
    tf.reset_default_graph()

    xs = tf.placeholder(tf.float32, [None, 50, 1600], name='xs')
    ys = tf.placeholder(tf.float32, [None, 2])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')
    x_image = tf.reshape(xs, [-1, 50, 1600, 1])

    net = Spec_CNN(x_image, training=is_training, keep_prob=keep_prob)

    W_fc = weights_variables([13 * 400 * 64, classes])
    b_fc = bias_variables([classes])
    pred = tf.matmul(net, W_fc) + b_fc
    prediction = tf.nn.softmax(pred, name='prediction')
    loss__ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=ys))

    learning_rate_base = 1e-4
    learning_rate_decay = 0.9
    learning_rate_step = 150
    global_step = tf.Variable(0)

    decaylearning_rate = tf.train.exponential_decay(learning_rate_base, global_step, learning_rate_step,
                                                    learning_rate_decay, staircase=True)
    train_step = tf.train.AdamOptimizer(decaylearning_rate).minimize(loss__)
    saver = tf.train.Saver()

    compound = component_seq

    # load data path
    path = u'../Matlab_data_training set/wav_data' + str(compound) + '/'

    datafile1 = path + str(compound) + 'component_data.mat'
    datafile2 = path + str(compound) + 'component_label_bc.mat'

    # save model path
    save_model_path = u'../Python_ConInceDeep_model/'
    save_file_dir = save_model_path + 'model_' + str(compound) + '/'
    save_file = save_file_dir + str(compound) + '/component.ckpt'

    if not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir)

    X = h5py.File(datafile1, 'r')
    X = np.transpose(X['wav_vir_data'])
    Xtrain = X[0:int(0.8 * X.shape[0])]
    Xvalid = X[int(0.8 * X.shape[0]):X.shape[0]]

    Y1 = h5py.File(datafile2, 'r')
    Y = Y1['label_bc']
    Y1 = Y
    Y2 = np.ones(Y1.shape) - Y1
    Y = np.concatenate((Y1, Y2), axis=1)
    Ytrain = Y[0:int(0.8 * Y.shape[0])]
    Yvalid = Y[int(0.8 * Y.shape[0]):Y.shape[0]]

    batch_size = 64
    epochs = 200
    batch_size_valid = 64

    acc_train_set = []
    loss_train_set = []
    acc_valid_set = []
    loss_valid_set = []

    acc_train_set.clear()
    loss_train_set.clear()
    acc_valid_set.clear()
    loss_valid_set.clear()

    max_acc = 0

    num_steps = Xtrain.shape[0] // batch_size
    num_step_valid = Xvalid.shape[0] // batch_size_valid
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            acc_train_total = 0
            loss_train_total = 0
            for step in range(num_steps):
                if step == num_steps:
                    batch_xs = Xtrain[step * batch_size: Xtrain.shape[0], :]
                    batch_ys = Ytrain[step * batch_size: Ytrain.shape[0], :]
                else:
                    batch_xs = Xtrain[step * batch_size: (step + 1) * batch_size, :]
                    batch_ys = Ytrain[step * batch_size: (step + 1) * batch_size, :]
                feed_dict = {xs: batch_xs, ys: batch_ys, keep_prob: 0.5, is_training: True}
                _, loss, predictions = sess.run([train_step, loss__, prediction], feed_dict=feed_dict)
                acc_train_sin = accuracy(predictions, batch_ys)
                acc_train_total += acc_train_sin
                loss_train_total += loss
            acc_train = acc_train_total / num_steps
            loss_train = loss_train_total / num_steps
            print('The epoch', epoch + 1, 'finished. The train accuracy %.4f%%' % acc_train, 'train loss =', loss_train,
                  ';', '\t', end=" ")

            acc_valid_total = 0
            loss_valid_total = 0
            for step in range(num_step_valid):
                if step == num_step_valid:
                    batch_xs_v = Xvalid[step * batch_size_valid: Xvalid.shape[0], :]
                    batch_ys_v = Yvalid[step * batch_size_valid: Yvalid.shape[0], :]
                else:
                    batch_xs_v = Xvalid[step * batch_size_valid: (step + 1) * batch_size_valid, :]
                    batch_ys_v = Yvalid[step * batch_size_valid: (step + 1) * batch_size_valid, :]
                feed_dict = {xs: batch_xs_v, ys: batch_ys_v, keep_prob: 1.0, is_training: False}
                _, loss_val, pred_val = sess.run([train_step, loss__, prediction], feed_dict=feed_dict)
                acc_val = accuracy(pred_val, batch_ys_v)
                acc_valid_total += acc_val
                loss_valid_total += loss_val
            acc_valid = acc_valid_total / (num_step_valid + 1)
            loss_valid = loss_valid_total / (num_step_valid + 1)
            print('The valid accuracy %.4f%%' % acc_valid, 'valid loss =', loss_valid)

            acc_train_set.append(acc_train)
            loss_train_set.append(loss_train)
            acc_valid_set.append(acc_valid)
            loss_valid_set.append(loss_valid)

            if epoch >= 20:
                last_five_acc_valid = acc_valid_set[-5:]
                if len(set(last_five_acc_valid)) == 1:
                    print('The acc_valid remained stable in 5 epochs, the training is completed.')
                    break

                if acc_valid > max_acc:
                    saver.save(sess, save_file)
                    print('Trained Model Saved.')
                    max_acc = acc_valid

    np.savetxt(save_file_dir + str(compound) + '/acc_train_set.txt', acc_train_set, fmt='%.2f')
    np.savetxt(save_file_dir + str(compound) + '/loss_train_set.txt', loss_train_set, fmt='%.2f')
    np.savetxt(save_file_dir + str(compound) + '/acc_valid_set.txt', acc_valid_set, fmt='%.2f')
    np.savetxt(save_file_dir + str(compound) + '/loss_valid_set.txt', loss_valid_set, fmt='%.2f')


# Input is the sequence number of the corresponding pure substance
single_train(3)


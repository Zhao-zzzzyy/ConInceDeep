
import tensorflow._api.v2.compat.v1 as tf
import numpy as np
import pandas as pd
import os
import scipy.io as matload
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.disable_v2_behavior()


def accuracy(predictions, labels):
    return round((100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0]), 5)


def single_test(kp, mix_data_path, root):
    mix_data_path = mix_data_path
    root = root
    names = os.listdir(root)

    # Load the mixture spectrum, its' labels and components information
    datafile1 = mix_data_path + '/true_mix_data.mat'
    true_mix_name_file = mix_data_path + '/mix_name'

    true_mix_name = matload.loadmat(true_mix_name_file)
    true_mix_name = true_mix_name['mix_name']

    Xtest = h5py.File(datafile1, 'r')
    Xtest = np.transpose(Xtest['true_mix_data'])

    Acetonitrile_label_file = mix_data_path + '/Acetonitrile_label.mat'
    Acetonitrile_label = matload.loadmat(Acetonitrile_label_file)
    Acetonitrile_label = Acetonitrile_label['acetonitrile_label']

    name_dic = {
                 '1': 'Diethyl malonate',
                 '2': 'Acetone',
                 '3': 'Acetonitrile',
                 '4': 'Ethanol',
                 '5': 'Diacetone alcohol',
                 '6': 'Cyclohexane'
                }

    label_dict = {
                   '3': Acetonitrile_label
                  }

    n = len(names)
    ypred = np.zeros((n * Xtest.shape[0], 2))
    list_dirs = os.walk(root)
    i = 0
    acc_test_set = []
    for root, dirs, files in list_dirs:
        for d in dirs:
            os.chdir(os.path.join(root, d))
            with tf.Session() as sess:
                new_saver = tf.train.import_meta_graph('./component.ckpt.meta')
                new_saver.restore(sess, "./component.ckpt")
                graph = tf.get_default_graph()
                xs = graph.get_operation_by_name('xs').outputs[0]
                keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
                is_training = graph.get_operation_by_name('is_training').outputs[0]
                prediction = graph.get_tensor_by_name('prediction:0')
                test_ypred = sess.run(prediction, feed_dict={xs: Xtest, keep_prob: kp, is_training: False})
            ypred[i * Xtest.shape[0]:(i + 1) * Xtest.shape[0], :] = test_ypred

            print('-------------------------------The result of: ', name_dic.get(d), '--------------------------------')
            print('compoent', i+1, ':', '------[', name_dic.get(d), ']------', 'finished.', 'The test accuracy %.4f%%' %
                  accuracy(ypred[i * Xtest.shape[0]:(i + 1) * Xtest.shape[0], :], label_dict.get(d)))
            acc_test_set.append(accuracy(ypred[i * Xtest.shape[0]:(i + 1) * Xtest.shape[0], :], label_dict.get(d)))

            # label to csv
            truelabel = pd.DataFrame(list(label_dict.get(d)))
            truelabel = truelabel[0]
            truelabel = truelabel.round(0)
            truelabel.rename = ['true label']
            predlabel = pd.DataFrame(list(ypred[i * Xtest.shape[0]:(i + 1) * Xtest.shape[0], :]))
            predlabel = predlabel[0]
            predlabel = predlabel.round(0)
            predlabel.rename = ['prediction label']
            true_mix_name = pd.DataFrame(true_mix_name)
            true_mix_name.rename = ['Mixture']

            label = pd.concat([true_mix_name, truelabel], axis=1)
            label = pd.concat([label, predlabel], axis=1)

            label_csv_path = u'../result_for_test/ConInceDeep/csv_label_true&pred'
            if not os.path.exists(label_csv_path):
                os.makedirs(label_csv_path)
            label_csvfile = label_csv_path + '/' + str(name_dic.get(d)) + '_label_true&pred.csv'
            label.to_csv(label_csvfile, encoding='utf_8_sig')

            # compute TruePosi(TP), FP, FN, TN
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            for index in range(label.shape[0]):
                if label.iat[index, 1] == 1:
                    if label.iat[index, 2] == 1:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if label.iat[index, 2] == 1:
                        FP += 1
                    else:
                        TN += 1
            print(name_dic.get(d), 'Confusion Matrix：TP=', TP, ', FN=', FN, ', FP=', FP, ', TN', TN, end='\n')
            precision = TP / (TP + FP)
            print(name_dic.get(d), 'Precision：Precision=TP/(TP+FP)=', precision, end='\n')
            recall = TP / (TP + FN)
            print(name_dic.get(d), 'Recall：Recall=TP/(TP+FN)=', recall, end='\n')
            F1_score = 2 * precision * recall / (precision + recall)
            print(name_dic.get(d), 'F1_score：F1_score=2*precision*recall/(precision+recall)=', F1_score, end='\n')
            i += 1
    print()

    return acc_test_set


kp = 1.0

print('The source of the testing mixture: 5-component mixtures', end="\n")
# the path of real mixture data as input
mix_data_path = u'../Matlab_data_testing set/5comps/'

if __name__ == '__main__':

    comp_seq = 3                      # modify the sequence number of pure substance model here
    print('ConInceDeep', end='\n')
    model_path = u'../Python_ConInceDeep_model/model_' + str(comp_seq)
    print(model_path, end='\n')

    test_set = single_test(kp, mix_data_path, model_path)

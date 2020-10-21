from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from .helper_func import get_class_md_combination
from ..utility import get_ordered_md


def split_data_set(down_sampled_file_path, n_min=10, sampling_percentage=0.1):
    """
    split down-sampled data (data set A in article) into training set and test set
    Test set data contains three parts:
    - all less than 10 molecules md_class;
    - all molecules in 10% random selected left classes which contain >= 10 molecules
    - 10% molecules in other classes ( stratified sampling )
    :param down_sampled_file_path: the result of down sampling (function down_sampling_mol)
        cid,md_class
        8873570,100110100
    :param n_min: the classes belong to test set which contain less than n_min molecules
    :param sampling_percentage:
    :return: training set and test set
    """
    cid2md_class = pd.read_csv(down_sampled_file_path, index_col='cid', dtype={'md_class': str})
    # cid2md_class = get_class_md_combination(cid2md_class, min_number=1)
    unique_classes = cid2md_class['md_class'].unique()

    # small classes
    small_class_bool = cid2md_class['md_class'].value_counts() < n_min
    small_class = small_class_bool[small_class_bool].index.to_list()
    print('There are {} small classes: {}'.format(len(small_class), small_class))

    unique_classes = list(set(unique_classes) - set(small_class))
    n_per = int(np.ceil(len(unique_classes) * sampling_percentage))
    sampled_class = np.random.choice(unique_classes, n_per, replace=False)
    unique_classes = list(set(unique_classes) - set(sampled_class))  # left 90%
    left_samples = cid2md_class[cid2md_class['md_class'].isin(unique_classes)].copy()
    split = StratifiedShuffleSplit(n_splits=1, test_size=sampling_percentage, random_state=42)
    train_set = pd.DataFrame()
    # test_set = pd.DataFrame()
    for train_inx, test_inx in split.split(left_samples, left_samples['md_class']):
        train_set = left_samples.iloc[train_inx]
        # test_set = left_samples.iloc[test_inx]
    test_set = cid2md_class[~cid2md_class.index.isin(train_set.index)].copy()
    test_set['type'] = '{}%_percentage_class'.format(sampling_percentage*100)
    test_set.loc[test_set['md_class'].isin(sampled_class), 'type'] = 'sampled_class'
    test_set.loc[test_set['md_class'].isin(small_class), 'type'] = 'small_class'
    return {'train_set': train_set, 'test_set': test_set}


def list2dic(a_list):
    a_dic = {}
    for i in a_list:
        a_dic[i] = 0
    return a_dic


def nn_model(x, y, result_dir):
    """
    model for multi-label classification
    :param x:
    :param y:
    :param result_dir:
    :return:
    """
    m_part1 = keras.Sequential([keras.layers.Dense(50, activation='selu', input_shape=[100]),
                                keras.layers.Dense(30, activation='selu')])
    m_part2 = keras.Sequential([
        keras.layers.Dense(50, activation='selu', input_shape=[30]),
        keras.layers.Dense(100, activation='selu'),
        keras.layers.Dense(8)])
    model = keras.Sequential([m_part1, m_part2])
    model.compile(optimizer='rmsprop', loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(x, y, epochs=10, batch_size=32)
    m_part1.save(os.path.join(result_dir, 'm_part1_2.h5'))
    model.save(os.path.join(result_dir, 'model_2.h5'))


def nn_model_regression(x, y, epochs, result_dir, callback=None,
                        frag_type=None, learning_rate=0.001):
    """
    Multivariate regression model
    :param x: molecular vectors, n x 100
    :param y: molecular descriptors, n x 9
    :param epochs:
    :param result_dir:
    :param callback: if using callback function
    :param frag_type: tandem or parallel
    :param learning_rate:
    :return:
    """
    m_part1 = keras.Sequential([
        keras.layers.Dense(50, activation='selu', input_shape=[x.shape[1]]),
        keras.layers.Dense(30, activation='selu')
        # keras.layers.Dropout(rate=0.05)
    ])
    m_part2 = keras.Sequential([
        keras.layers.Dense(50, activation='selu', input_shape=[30]),
        # keras.layers.Dropout(rate=0.1),
        keras.layers.Dense(100, activation='selu'),
        # keras.layers.Dropout(rate=0.05),
        # non-negative by softplus
        keras.layers.Dense(y.shape[1], activation='softplus')])
    model = keras.Sequential([m_part1, m_part2])
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    # opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mse', metrics=['mse'])

    if callback:
        # Stop training when a monitored quantity has stopped improving.
        early_stopping_callback = keras.callbacks.EarlyStopping(
            patience=30, monitor='val_loss', mode='min',
            restore_best_weights=False)
        history = model.fit(x, y, epochs=epochs, batch_size=64,
                            validation_split=0.2, verbose=2,
                            callbacks=[early_stopping_callback])

    else:
        history = model.fit(x, y, epochs=epochs, batch_size=64,
                            verbose=2, validation_split=0.2)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.to_csv(os.path.join(result_dir, 'history_reg_{}.csv'.format(frag_type)))
    m_part1.save(os.path.join(result_dir, 'm_part1_reg_{}.h5'.format(frag_type)))
    model.save(os.path.join(result_dir, 'model_reg_{}.h5'.format(frag_type)))
    return m_part1


def predict(model_file_path, test_data_file_path, mol2vec_file_path):
    """
    Perform prediction with a pre-trained model
    :param model_file_path:
    :param test_data_file_path:
    :return:
    """
    # Load signature genes and celltype labels
    # sig_genes = pd.read_table(self.model_dir + "/genes.txt", index_col=0)
    # self.sig_genes = list(sig_genes['0'])
    model = keras.models.load_model(model_file_path)
    # Predict using loaded model
    test_data = pd.read_csv(test_data_file_path, index_col=0)
    mol2vec = pd.read_csv(mol2vec_file_path, index_col=0, header=None)
    mol2vec_test_data = mol2vec.loc[test_data.index, :]
    predictions = model.predict(mol2vec_test_data)
    md = get_ordered_md()
    pred_df = pd.DataFrame(predictions, index=test_data.index, columns=md)
    # pred_df.to_csv(out_name, sep="\t")
    pred_df = pred_df.round(0).astype(int)
    pred_df = get_class_md_combination(pred_df, min_number=0)
    return pred_df


def evaluate(y_pred, y_true, model_name=None):
    """

    :param y_pred: dataframe
    :param y_true: dataframe
    :return:
    """
    class2count = {}
    n_total = 'n_total_{}'.format(model_name)
    n_correct = 'n_correct_{}'.format(model_name)
    for cid in y_true.index:
        # md_class = y_true.loc[cid, 'md_class']
        current_y_true = y_true.loc[cid, 'md_class']
        current_y_pred = y_pred.loc[cid, 'md_class']
        if current_y_true not in class2count:
            class2count[current_y_true] = {n_total: 0, n_correct: 0}
        class2count[current_y_true][n_total] += 1
        if current_y_pred == current_y_true:
            class2count[current_y_true][n_correct] += 1
    result = pd.DataFrame.from_dict(data=class2count, orient='index')
    result['pred_acc_{}'.format(model_name)] = result[n_correct] / result[n_total]
    return result


def load_pre_trained_model(model_fp):
    return keras.models.load_model(model_fp)


if __name__ == '__main__':
    pass

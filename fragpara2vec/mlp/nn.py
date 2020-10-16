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


def nn_model_regression(x, y, epochs, result_dir, callback=None):
    """
    Multivariate regression model
    model checkpoint: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
    :param x:
    :param y:
    :param result_dir:
    :param callback: if using callback function
    :return:
    """
    m_part1 = keras.Sequential([keras.layers.Dense(50, activation='selu', input_shape=[x.shape[1]]),
                                keras.layers.Dense(30, activation='selu'),
                                keras.layers.Dropout(rate=0.1)])
    m_part2 = keras.Sequential([
        keras.layers.Dense(50, activation='selu', input_shape=[30]),
        keras.layers.Dropout(rate=0.1),
        keras.layers.Dense(100, activation='selu'),
        keras.layers.Dropout(rate=0.1),
        keras.layers.Dense(y.shape[1], activation='softplus')])  # non-negative
    model = keras.Sequential([m_part1, m_part2])
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])

    if callback:
        # Model weights are saved at the end of every epoch, if it's the best seen
        # so far.
        # checkpoint_filepath = os.path.join(result_dir, 'checkpoint', 'chp')
        # if not os.path.exists(checkpoint_filepath):
        #     os.makedirs(checkpoint_filepath)
        early_stopping_callback = keras.callbacks.EarlyStopping(
            patience=10,
            monitor='val_loss',
            mode='min',
            restore_best_weights=False)
        history = model.fit(x, y, epochs=epochs, batch_size=32, verbose=2,
                            validation_split=0.2, callbacks=[early_stopping_callback])

    else:
        history = model.fit(x, y, epochs=epochs, batch_size=32, verbose=2, validation_split=0.2)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.to_csv(os.path.join(result_dir, 'history_reg.csv'))
    m_part1.save(os.path.join(result_dir, 'm_part1_reg.h5'))
    model.save(os.path.join(result_dir, 'model_reg.h5'))
    return m_part1


if __name__ == '__main__':
    pass

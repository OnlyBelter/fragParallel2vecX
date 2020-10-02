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


def sampling_train_set(mol2md_info_fn, n_max=5000):
    """
    select a subset of training set without some classes (remove 20%)
    :param mol2md_info_fn:
    :param n_max: max samples of sampling in each class
    :return: sampling result, a list of cid
    """
    mol2md_info = pd.read_csv(mol2md_info_fn, index_col='cid')
    mol2md_info = get_class_md_combination(mol2md_info, min_number=1)
    unique_labels = mol2md_info['class'].unique()
    n_80_per = int(np.ceil(len(unique_labels) * 0.9))
    unique_labels_80 = np.random.choice(unique_labels, n_80_per, replace=False)
    small_class_bool = mol2md_info['class'].value_counts() < 10
    small_class = small_class_bool[small_class_bool].index.to_list()
    print('num: {}, small class: {}'.format(len(small_class), small_class))
    unique_labels_80 = set(unique_labels_80) - set(small_class)
    selected_mol2md_info = mol2md_info[mol2md_info['class'].isin(unique_labels_80)].copy()
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_set = pd.DataFrame()
    test_set = pd.DataFrame()
    for train_inx, test_inx in split.split(selected_mol2md_info, selected_mol2md_info['class']):
        train_set = selected_mol2md_info.iloc[train_inx]
        test_set = selected_mol2md_info.iloc[test_inx]
    other_mol2md_info = mol2md_info[~mol2md_info['class'].isin(unique_labels_80)].copy()
    test_set = other_mol2md_info.append(test_set)
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
                                keras.layers.Dense(30, activation='selu')])
    m_part2 = keras.Sequential([
        keras.layers.Dense(50, activation='selu', input_shape=[30]),
        keras.layers.Dense(100, activation='selu'),
        keras.layers.Dense(y.shape[1], activation='softplus')])  # non-negative
    model = keras.Sequential([m_part1, m_part2])
    model.compile(optimizer='rmsprop', loss='mse',
                  metrics=['mse'])

    if callback:
        # Model weights are saved at the end of every epoch, if it's the best seen
        # so far.
        checkpoint_filepath = os.path.join(result_dir, 'checkpoint', 'chp')
        if not os.path.exists(checkpoint_filepath):
            os.makedirs(checkpoint_filepath)
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_mse',
            mode='max',
            save_best_only=True)
        history = model.fit(x, y, epochs=epochs, batch_size=32, verbose=2, validation_split=0.2,
                            callbacks=[model_checkpoint_callback])

        # The model weights (that are considered the best) are loaded into the model.
        model.load_weights(checkpoint_filepath)
        m_part1.load_weights(checkpoint_filepath)
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

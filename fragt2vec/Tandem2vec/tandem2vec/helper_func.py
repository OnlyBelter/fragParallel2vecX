import timeit
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from gensim.models import fasttext, word2vec
from gensim.test.utils import get_tmpfile
from ...utility import SELECTED_MD, PRMIER_NUM


def get_class(frag_info, selected_md=None, min_number=3):
    """
    get unique class depends on different molecular descriptors
    frag_info: a dataframe which contains fragment smiles, selected_md
    selected_md: selected molecular descriptors
    min_number: the minimal number of fragment in each class
    :return: fragment, class(the product of multiple primer numbers), class_id(0 to n), class_num(count each class)
    """
    if not selected_md:
        selected_md = SELECTED_MD
    md_num = len(selected_md)
    if md_num <= len(PRMIER_NUM):
        unique_code = PRMIER_NUM[:md_num]
    else:
        raise Exception('Please give more primer number to PRMIER_NUM...')
    # frag_info = frag_info.set_index('fragment')
    frag_info = frag_info.loc[:, selected_md].copy()
    frag_info[frag_info >= 1] = 1
    frag_info = frag_info.apply(lambda x: np.multiply(x, unique_code), axis=1)
    frag_info[frag_info == 0] = 1
    frag2class = frag_info.apply(lambda x: mul_list(x), axis=1)
    frag2class = pd.DataFrame(frag2class, columns=['class'])

    frag_class2num = {}
    for c in frag2class.index:
        class_num = frag2class.loc[c, 'class']
        if class_num not in frag_class2num:
            frag_class2num[class_num] = 0
        frag_class2num[class_num] += 1
    frag_class2num_df = pd.DataFrame.from_dict(frag_class2num, orient='index', columns=['class_num'])
    frag2class = frag2class.merge(frag_class2num_df, left_on='class', right_index=True)
    frag2class = frag2class[frag2class['class_num'] >= min_number].copy()
    print('  >the shape of frag2class after filtered: {}'.format(frag2class.shape))

    unique_class = sorted(frag2class['class'].unique())
    code2id = {unique_class[i]: i for i in range(len(unique_class))}
    print(code2id)
    frag2class['class_id'] = frag2class['class'].apply(lambda x: code2id[x])

    # depth = len(code2id)
    # y_one_hot = tf.one_hot(frag2class_filtered.class_id.values, depth=depth)
    # print('  >the shape of one hot y: {}'.format(y_one_hot.shape))
    return frag2class


def mul_list(a_list):
    result = 1
    for i in a_list:
        result *= i
    return result


def vis_class(X, labels, title, file_path=None):
    """
    plot reduced fragment vector
    :param X: two dimensions np.array
    :param labels:
    :param title:
    :param file_path:
    :return:
    """
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    plt.figure(figsize=(15, 12))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14, label=k)
        plt.text(xy[0, 0], xy[0, 1], str(k), fontsize=18)

    #         xy = X[class_member_mask & ~core_samples_mask]
    #         plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #                  markeredgecolor='k', markersize=6, label=k)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if file_path:
        plt.savefig(file_path, dpi=300)


def get_xy(frag2vec, frag2class, d=100):
    """
    merge frag2vec and frag_info, then get vector and one-hot y
    :param: d, dimension of fragment vector
    """
    # frag2vec = pd.read_csv(frag2vec_fp, index_col='frag_id')
    # frag_info = pd.read_csv(frag_info_fp, index_col='fragment')
    frag2class = frag2class.loc[:, ['class']].copy()
    class_id = np.unique(frag2class.values)
    print('there are {} unique classes: {}'.format(len(np.unique(frag2class.values)), class_id))
    class2inx = {class_id[i]: i for i in range(len(class_id))}
    print(class2inx)
    frag2vec = frag2vec.merge(frag2class, left_index=True, right_index=True)
    depth = len(class_id)
    y_inx = frag2vec.loc[:, ['class']].apply(lambda x: class2inx[x.values[0]], axis=1)
    # print(y_inx, type(y_inx))
    y = tf.one_hot(y_inx.values, depth=depth)
    print('  >the first 2 classis {}'.format(frag2vec.loc[:, ['class']].head(2)))
    return {'x': frag2vec.iloc[:, range(d)], 'y': y}


def train_fasttext_model(infile_name, outfile_name=None, dim=100, ws=4, min_count=3, n_jobs=1,
                         minn=1, maxn=2, method='cbow', epoch=30):
    """
    training fasttext (Tandem2vec) model on corpus file extracted from molecules

    - parameters in FastText
    https://fasttext.cc/docs/en/options.html

    - parameters in gensim
    https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/FastText_Tutorial.ipynb

    - parameters of fasttext in gensim vs original FastText

        sg=0 means using 'cbow' model,
        size means dim, window means ws, iter means epoch,
        min_count means minCount, min_n means minn, max_n means maxn

    :param infile_name: Path to the file on disk, a file that contains sentences(one line = one sentence).
           Words must be already preprocessed and separated by whitespace.
    :param outfile_name:
    :param dim: size of word vectors [100]
    :param ws: size of the context window [4]
    :param min_count: minimal number of word occurrences [3]
    :param n_jobs:
    :param minn: min length of char ngram [1]
    :param maxn: max length of char ngram [2]
    :param method: skip-gram / cbow [cbow]
    :param epoch: number of epochs [30]
    :return: fasttext model
    """

    if method.lower() == 'skip-gram':
        sg = 1
    elif method.lower() == 'cbow':
        sg = 0
    else:
        raise ValueError('skip-gram or cbow are only valid options')

    start = timeit.default_timer()
    model = fasttext.FastText(sg=sg, size=dim, window=ws,
                              min_count=min_count, min_n=minn, max_n=maxn, workers=n_jobs)
    # model = word2vec.Word2Vec(corpus, size=vector_size, window=window, min_count=min_count, workers=n_jobs, sg=sg,
    #                           **kwargs)
    # corpus = word2vec.LineSentence(infile_name)
    print('>>> Start to read molecular sentences...')
    model.build_vocab(corpus_file=infile_name)
    print('Count of molecular sentences: {}, count of unique fragment: {}'.format(model.corpus_count, len(model.wv.vocab)))
    print('>>> Start to training model...')
    model.train(corpus_file=infile_name, total_examples=model.corpus_count,
                epochs=epoch, total_words=len(model.wv.vocab))
    if outfile_name:
        # fname = get_tmpfile("fasttext.model")
        model.save(outfile_name)

    stop = timeit.default_timer()
    print('Runtime: ', round((stop - start) / 60, 2), ' minutes')
    return model

"""
step3: Training molFrag2vec model by FastText, and get the vectors of all fragments.
https://radimrehurek.com/gensim/models/fasttext.html
# usage: without supervised training
$ python step3_training_frag_vec_model.py big-data/moses_dataset/result/step2_parallel_frag_smiles_sentence.csv big-data/moses_dataset/result/
 --model_fn step3_model_parallel2vec.bin --frag_vec_fn step3_model_parallel2vec.csv

# usage: with supervised training
$ python step3_training_frag_vec_model.py big-data/moses_dataset/result/step2_parallel_frag_smiles_sentence.csv big-data/moses_dataset/result/
 --model_fn step3_model_parallel2vec.bin --frag_vec_fn step3_model_parallel2vec.csv --supervised_training
"""
from gensim.models import fasttext
# import fasttext
import argparse
import pandas as pd
import os
import csv
from ...utility import cal_md_by_smiles, get_format_time
# from pub_func import get_format_time
from ..classifier import SuperviseClassModel
from .helper_func import get_class, vis_class, get_xy, train_fasttext_model


def get_frag_vector(model_fp, frag_id2vec_fp, frag_smiles=('all',)):
    """
    get fragment vector from pre-trained model
    https://radimrehurek.com/gensim/models/fasttext.html#module-gensim.models.fasttext
    :param model_fp: file path of pre-trained model
    :param frag_id2vec_fp: file path of frag_id2vec
    :param frag_smiles: list
           get fragment vector of the fragments in this list, get all fragment vectors in model if ('all',)
    :return:
    """
    model = fasttext.FastText.load(model_fp)
    words = model.wv.vocab
    # frag2vec = {}
    # for f in words:
    #     print(type(model.wv))
    #     # model.wv.vectors[1]
    #     frag2vec[f] = model.wv(f)
    frag2vec_df = pd.DataFrame(model.wv.vectors, index=model.wv.index2word)
    if len(frag_smiles) == 1 and frag_smiles[0] == 'all':
        pass
    elif len(frag_smiles) >= 1:
        frag2id = {}
        for smiles in frag_smiles:
            if smiles in words:
                frag2id[smiles] = words[smiles].index
            else:
                print('>>> Fragment SMILES {} does not exist in the model.'.format(smiles))
        frag2vec_df = frag2vec_df.loc[list(frag2id.keys()), :].copy()
    if frag_id2vec_fp:
        print('>>> There are {} fragments were returned.'.format(frag2vec_df.shape[0]))
        frag2vec_df.to_csv(frag_id2vec_fp, index_label='fragment')
    else:
        return frag2vec_df


def get_mol_vector(model_fp, frag_id2vec_fp, ):
    pass


def get_mol_vec(frag2vec, data_set, result_path):
    """
    sum all fragment vector to get molecule vector
    :param frag2vec:
    :param data_set: step5_x_training_set.csv
    :return:
    """
    frag2vec_df = pd.read_csv(frag2vec, index_col=0)
    cid2vec = {}
    counter = 0
    with open(data_set, 'r') as handle:
        train_set_reader = csv.reader(handle, delimiter=',')
        for row in train_set_reader:
            if row[-1] != '0':
                cid, mol_path, mol_inx, frag_smiles = row
                frags = frag_smiles.split(' ')
                try:
                    cid2vec[cid] = frag2vec_df.loc[frags, :].sum().values
                except KeyError:
                    print('fragments {} are not in lib'.format(frag_smiles))
                if len(cid2vec) == 500000:
                    pid2vec_df = pd.DataFrame.from_dict(cid2vec, orient='index')
                    pid2vec_df.to_csv(result_path, mode='a', header=False, float_format='%.3f')
                    cid2vec = {}
            if counter % 10000 == 0:
                print('>>> Processing line {}...'.format(counter))
            counter += 1
    # the last part
    pid2vec_df = pd.DataFrame.from_dict(cid2vec, orient='index')
    pid2vec_df.to_csv(result_path, mode='a', header=False, float_format='%.3f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training molFrag2vec model using FastText')
    parser.add_argument('input_fn', help='training set file path')
    parser.add_argument('result_dir', help='the directory of result files')
    parser.add_argument('--model_fn', help='where to save trained model')
    parser.add_argument('--frag_vec_fn', help='where to save fragment vector')
    # parser.add_argument('--frag2num_fp', help='the file path of frag2num from step2', default='no_input')
    parser.add_argument('--supervised_training', action='store_true', default=False,
                        help='if train fragment vectors by supervised method to reorganize vector space.')
    args = parser.parse_args()
    input_file = args.input_fn
    result_dir = args.result_dir
    model_fn = args.model_fn
    frag_vec_fn = args.frag_vec_fn
    supervised_training = args.supervised_training
    # frag2num_fp = args.frag2num_fp
    model_name = model_fn.replace('.csv', '').split('_')[-1]
    model_fp = os.path.join(result_dir, model_fn)
    frag2vec_fp = os.path.join(result_dir, frag_vec_fn)

    t0 = get_format_time()
    print('  >Start to train vector model in {}...'.format(t0))
    train_fasttext_model(input_file, model_fp, n_jobs=5)
    # mol2vec_fp = os.path.join(result_dir, 'selected_mol2vec.csv')
    get_frag_vector(model_fp, frag_id2vec_fp=frag2vec_fp)
    t1 = get_format_time()
    print('  >Finished training vector model in {}...'.format(t1))

    if supervised_training:
        # if frag2num_fp == 'no_input':
        #     raise Exception('You must give the file path of file frag2num from step2 by parameter --frag2num_fp, '
        #                     'since supervised_training is open.')
        # get fragment information
        print('  >Start to get supervise-trained fragment vectors...')
        frag2vec = pd.read_csv(frag2vec_fp, index_col='fragment')
        frag_smiles_list = frag2vec.index.to_list()
        frag_smiles_list = [i for i in frag_smiles_list if i != '</s>']  # remove non-SMILES holder
        frag_info = cal_md_by_smiles(smiles_list=frag_smiles_list)

        frag_info.to_csv(os.path.join(result_dir, 'step3_model_{}_frag_info.csv'.format(model_name)),
                         index_label='fragment')
        # x = frag2vec.values
        frag2class = get_class(frag_info, min_number=1)
        xy_result = get_xy(frag2vec=frag2vec, frag2class=frag2class)
        x = xy_result['x']
        y = xy_result['y']  # one hot array
        n_class = y.shape[1]
        print('  >number of class is: {}'.format(n_class))
        print('  > type of x: {}'.format(type(x)))
        # training fragment vector
        supervised_model = SuperviseClassModel(n_output=n_class)
        supervised_model.model_compile()
        supervised_model.training(x=x, y=y)
        frag2vec_new = supervised_model.get_embedding_vec(x)
        frag2vec_new_df = pd.DataFrame(data=frag2vec_new, index=x.index)
        frag2vec_new_df.to_csv(os.path.join(result_dir, 'step3_model_{}_supervise_trained.csv'.format(model_name)))

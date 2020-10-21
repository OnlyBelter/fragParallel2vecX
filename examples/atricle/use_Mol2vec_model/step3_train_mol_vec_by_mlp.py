import os
import pandas as pd
from fragpara2vec.utility import get_ordered_md
from fragpara2vec.mlp import nn_model_regression


if __name__ == '__main__':
    root_dir = r'F:\github\fragParallel2vecX\big_data'
    subdir = '08_train_mol_vec_by_mlp'
    result_dir = os.path.join(root_dir, subdir)
    selected_mol2md_file_path = os.path.join(result_dir, 'selected_cid2md.csv')  # y
    train_set_file_path = os.path.join(result_dir, 'train_set.csv')
    test_set_file_path = os.path.join(result_dir, 'test_set.csv')
    frag_type = 'Mol2vec'  # Mol2vec model

    # train model
    print('Start to train MLP model...')
    train_set = pd.read_csv(train_set_file_path, index_col='cid')
    selected_mol2md = pd.read_csv(selected_mol2md_file_path, index_col='cid')
    md = get_ordered_md()
    selected_mol2md = selected_mol2md.loc[:, md].copy()
    print('>>> training model of {}'.format(frag_type))
    _result_dir = os.path.join(result_dir, 'pre_trained_model', frag_type)
    if not os.path.exists(_result_dir):
        os.makedirs(_result_dir)
    mol_vec_file_path = os.path.join(result_dir, 'fingerprinting', 'mol_vec_{}_frag.csv'.format(frag_type))
    mol_vec = pd.read_csv(mol_vec_file_path, index_col=0, header=None)
    x = mol_vec.loc[mol_vec.index.isin(train_set.index), :]
    y = selected_mol2md.loc[selected_mol2md.index.isin(train_set.index)]
    nn_model_regression(x=x, y=y, epochs=1000, callback=True,
                        result_dir=_result_dir, frag_type=frag_type)

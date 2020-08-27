import os
import pandas as pd
from mordred import descriptors
from fragt2vec.Tandem2vec import tandem2vec
from fragt2vec.utility import cal_md_by_smiles


def get_md(smiles):
    """
    calculate MD of fragment
    :param smiles:
    :return: ['MW', 'SLogP', 'nRot'] values, a dataframe
    """
    md_list = ['MW', 'SLogP', 'nRot', 'nN', 'nS', 'nO', 'nX', 'nBondsD', 'nBondsT', 'naRing', 'nARing']
    desc = [descriptors.Weight, descriptors.RotatableBond, descriptors.SLogP,
            descriptors.AtomCount, descriptors.BondCount, descriptors.RingCount]
    md_values = cal_md_by_smiles(smiles, md_list, desc=desc)
    return md_values


if __name__ == '__main__':
    root_dir = '../../big_data/'
    # mol_sentence_file_name = 'step2_parallel_frag_smiles_sentence.csv'

    # sub_dir1 = '03_fragment'
    sub_dir2 = '05_model_Tandem2vec'
    minn = 1
    maxn = 2
    mol_sentence_file_name = 'frag_id2vec_minn_{}_maxn_{}.csv'.format(minn, maxn)
    model_file_name = 'tandem2vec_model_minn_{}_maxn_{}.bin'.format(minn, maxn)
    # model_file_name = 'tandem2vec_model.bin'
    frag_id2vec_file_path = os.path.join(root_dir, sub_dir2, mol_sentence_file_name)

    # # get the vector of fragments from pre-trained model
    tandem2vec.get_frag_vector(frag_id2vec_fp=frag_id2vec_file_path,
                               model_fp=os.path.join(root_dir, sub_dir2, model_file_name))

    # calculate molecular descriptor(MD) of fragments
    frag_id2vec = pd.read_csv(frag_id2vec_file_path, index_col=0)
    frag_smiles = frag_id2vec.index.to_list()
    frag_smiles = [i for i in frag_smiles if i != 'UNK']  # remove rare fragment token
    md = get_md(frag_smiles)
    md.to_csv(os.path.join(root_dir, sub_dir2, 'frag_smiles2md.csv'), index_label='fragment')

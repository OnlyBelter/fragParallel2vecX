"""
train fragments from tandem method to get vector
"""
import os
from fragt2vec import tandem2vec


def run_demo():
    root_dir = '../../big_data/'
    # mol_sentence_file_name = 'step2_parallel_frag_smiles_sentence.csv'
    mol_sentence_file_name = 'frag_smiles_sentence_demo_replace_uncommon_frag.txt'
    sub_dir1 = '03_fragment'
    sub_dir2 = '05_model_Tandem2vec'
    model_file_name = 'tandem2vec_model.bin'
    tandem2vec.train_fasttext_model(infile_name=os.path.join(root_dir, sub_dir1, mol_sentence_file_name),
                                    outfile_name=os.path.join(root_dir, sub_dir2, model_file_name),
                                    n_jobs=4, epoch=5, min_count=3)


if __name__ == '__main__':
    root_dir = '../../big_data/'
    # mol_sentence_file_name = 'step2_parallel_frag_smiles_sentence.csv'
    mol_sentence_file_name = 'step2_parallel_frag_smiles_sentence_replace_uncommon_frag.csv'

    sub_dir1 = '03_fragment'
    sub_dir2 = '05_model_Tandem2vec'
    minn = 0
    maxn = 0
    model_file_name = 'tandem2vec_model_minn_{}_maxn_{}.bin'.format(minn, maxn)
    tandem2vec.train_fasttext_model(infile_name=os.path.join(root_dir, sub_dir1, mol_sentence_file_name),
                                    outfile_name=os.path.join(root_dir, sub_dir2, model_file_name),
                                    n_jobs=4, epoch=5, min_count=3, minn=minn, maxn=maxn)

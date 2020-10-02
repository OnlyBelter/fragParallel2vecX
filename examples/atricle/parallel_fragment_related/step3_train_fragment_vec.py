"""
train fragments from tandem method to get vector
"""
import os
from fragpara2vec import parallel2vec
from fragpara2vec.Mol2vec import train_word2vec_model


def run_demo():
    root_dir = '../../../big_data/'
    # mol_sentence_file_name = 'step2_parallel_frag_smiles_sentence.csv'
    mol_sentence_file_name = 'frag_smiles_sentence_demo_replace_uncommon_frag.txt'
    sub_dir1 = '03_fragment'
    sub_dir2 = '05_model_Tandem2vec'
    model_file_name = 'tandem2vec_model.bin'
    parallel2vec.train_fasttext_model(infile_name=os.path.join(root_dir, sub_dir1, mol_sentence_file_name),
                                      outfile_name=os.path.join(root_dir, sub_dir2, model_file_name),
                                      n_jobs=4, epoch=5, min_count=3)


if __name__ == '__main__':
    root_dir = '../../../big_data/'
    # mol_sentence_file_name = 'step2_parallel_frag_smiles_sentence.csv'
    frag_sentence_type = 'tandem'  # parallel or tandem
    mol_sentence_file_name = '{}_frag_smiles_sentence_replaced_uncommon_frag.csv'.format(frag_sentence_type)
    sub_dir1 = '03_fragment'
    if frag_sentence_type == 'parallel':
        sub_dir2 = '06_model_Parallel2vec'
    else:
        sub_dir2 = '05_model_Tandem2vec'
    # 0 & 0, or 1 & 2
    minn = 0
    maxn = 0
    model_file_name = '{}2vec_model_minn_{}_maxn_{}.bin'.format(frag_sentence_type, minn, maxn)
    fasttest_model_fp = os.path.join(root_dir, sub_dir2, model_file_name)
    if not os.path.exists(fasttest_model_fp):
        parallel2vec.train_fasttext_model(infile_name=os.path.join(root_dir, sub_dir1, mol_sentence_file_name),
                                          outfile_name=fasttest_model_fp,
                                          n_jobs=6, epoch=5, min_count=5, minn=minn, maxn=maxn,
                                          dim=100, method='cbow', ws=4)

"""
train fragments from tandem method to get vector
"""
import os
from fragpara2vec.Parallel2vec.parallel2vec import train_fasttext_model, get_frag_vector_fasttext
# from fragpara2vec.Mol2vec import train_word2vec_model


def run_demo():
    root_dir = '../../../big_data/'
    # mol_sentence_file_name = 'step2_parallel_frag_smiles_sentence.csv'
    mol_sentence_file_name = 'frag_smiles_sentence_demo_replace_uncommon_frag.txt'
    sub_dir1 = '03_fragment'
    sub_dir2 = '05_model_Tandem2vec'
    model_file_name = 'tandem2vec_model.bin'
    train_fasttext_model(infile_name=os.path.join(root_dir, sub_dir1, mol_sentence_file_name),
                         outfile_name=os.path.join(root_dir, sub_dir2, model_file_name),
                         n_jobs=4, epoch=5, min_count=3)


if __name__ == '__main__':
    root_dir = '../../../big_data/'
    # mol_sentence_file_name = 'step2_parallel_frag_smiles_sentence.csv'
    for frag_sentence_type in ['tandem', 'parallel']:
        # frag_sentence_type = 'tandem'  # parallel or tandem
        mol_sentence_file_name = '{}_frag_smiles_sentence_replaced_uncommon_frag.csv'.format(frag_sentence_type)
        sub_dir1 = '03_fragment'
        if frag_sentence_type == 'parallel':
            sub_dir2 = '06_model_Parallel2vec'
        else:
            sub_dir2 = '05_model_Tandem2vec'
        for minn, maxn in [(0, 0), (1, 2)]:  # [(0, 0), (1, 2)]
            # 0 & 0, or 1 & 2
            print('Training model {}, ({}, {})'.format(frag_sentence_type, minn, maxn))
            model_file_name = '{}2vec_model_minn_{}_maxn_{}.bin'.format(frag_sentence_type, minn, maxn)
            fasttest_model_fp = os.path.join(root_dir, sub_dir2, model_file_name)
            if not os.path.exists(fasttest_model_fp):
                train_fasttext_model(infile_name=os.path.join(root_dir, sub_dir1, mol_sentence_file_name),
                                     outfile_name=fasttest_model_fp,
                                     n_jobs=7, epoch=100, min_count=5, minn=minn, maxn=maxn,
                                     dim=100, method='skip-gram', ws=5)

    # get fragment vector
    for frag_sentence_type in ['tandem', 'parallel']:
        print('Deal with fragment sentence type: {}'.format(frag_sentence_type))
        # frag_sentence_type = 'tandem'  # parallel or tandem
        if frag_sentence_type == 'parallel':
            sub_dir2 = '06_model_Parallel2vec'
        else:
            sub_dir2 = '05_model_Tandem2vec'
        # sub_dir_tandem = '05_model_Tandem2vec'
        # sub_dir_parallel = '06_model_Parallel2vec'
        # minn = 1
        # maxn = 2
        for minn, maxn in [(0, 0), (1, 2)]:  # [(0, 0), (1, 2)]
            print('>>> Deal with minn: ({}), maxn: ({})'.format(minn, maxn))
            frag2vec_file_name = 'frag_smiles2vec_minn_{}_maxn_{}_{}.csv'.format(minn, maxn, frag_sentence_type)
            model_file_name = '{}2vec_model_minn_{}_maxn_{}.bin'.format(frag_sentence_type, minn, maxn)
            # model_file_name = 'tandem2vec_model.bin'
            frag_smiles2vec_file_path = os.path.join(root_dir, sub_dir2, frag2vec_file_name)

            # get the vector of fragments from pre-trained model by fasttext
            if not os.path.exists(frag_smiles2vec_file_path):
                get_frag_vector_fasttext(frag_id2vec_fp=frag_smiles2vec_file_path,
                                         model_fp=os.path.join(root_dir, sub_dir2, model_file_name))

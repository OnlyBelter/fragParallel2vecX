import os
import pandas as pd
from mordred import descriptors
from fragpara2vec.Parallel2vec import get_frag_vector_fasttext
from fragpara2vec.utility import cal_md_by_smiles, SELECTED_MD
# from fragpara2vec.Mol2vec import get_frag_vector_word2vec


# def get_md(smiles):
#     """
#     calculate MD of fragment
#     :param smiles:
#     :return: ['MW', 'SLogP', 'nRot'] values, a dataframe
#     """
#     md_list = ['MW', 'SLogP', 'nRot'] + SELECTED_MD
#     desc = [descriptors.Weight, descriptors.RotatableBond, descriptors.SLogP,
#             descriptors.AtomCount, descriptors.BondCount, descriptors.RingCount]
#     md_values = cal_md_by_smiles(smiles, md_list, desc=desc)
#     return md_values


if __name__ == '__main__':
    root_dir = '../../../big_data/'
    # mol_sentence_file_name = 'step2_parallel_frag_smiles_sentence.csv'

    # sub_dir1 = '03_fragment'
    # sub_dir_mol2vec = '04_model_Mol2vec'
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
        for minn, maxn in [(0, 0), (1, 2)]:
            print('>>> Deal with minn: ({}), maxn: ({})'.format(minn, maxn))
            frag2vec_file_name = 'frag_smiles2vec_minn_{}_maxn_{}_{}.csv'.format(minn, maxn, frag_sentence_type)
            model_file_name = '{}2vec_model_minn_{}_maxn_{}.bin'.format(frag_sentence_type, minn, maxn)
            # model_file_name = 'tandem2vec_model.bin'
            frag_smiles2vec_file_path = os.path.join(root_dir, sub_dir2, frag2vec_file_name)

            # get the vector of fragments from pre-trained model by fasttext
            if not os.path.exists(frag_smiles2vec_file_path):
                get_frag_vector_fasttext(frag_id2vec_fp=frag_smiles2vec_file_path,
                                         model_fp=os.path.join(root_dir, sub_dir2, model_file_name))

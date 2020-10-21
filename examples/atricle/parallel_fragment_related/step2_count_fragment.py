import os
import re
import fragpara2vec
import pandas as pd
from gensim.test.utils import get_tmpfile
from fragpara2vec.utility import count_frag_in_mol_sentence
from fragpara2vec.utility import insert_unk, cal_md_by_smiles
import pkg_resources


def run_demo():
    demo_dir = pkg_resources.resource_filename('fragpara2vec', 'demo_data')
    my_dir = r'F:\tmp'
    # mol_sentence_file_name = 'step2_parallel_frag_smiles_sentence.csv'
    mol_sentence_file_name = 'step2_parallel_frag_smiles_sentence.csv'
    # mol_sf_replace_uncommon_frag = mol_sentence_file_name.replace('.csv', '_replace_uncommon_frag.csv')
    mol_sf_replace_uncommon_frag = re.sub(r'(\..*)', r'_replace_uncommon_frag\g<1>', mol_sentence_file_name)
    sub_dir1 = '03_fragment'
    insert_unk(corpus=os.path.join(demo_dir, sub_dir1, mol_sentence_file_name),
               out_corpus=os.path.join(my_dir, sub_dir1, mol_sf_replace_uncommon_frag),
               threshold=2)
    frag2count = count_frag_in_mol_sentence(os.path.join(my_dir, sub_dir1, mol_sf_replace_uncommon_frag))
    frag2count.to_csv(os.path.join(my_dir, sub_dir1, 'frag2count_' + mol_sentence_file_name.replace('txt', 'csv')),
                      index_label='fragment')


if __name__ == '__main__':
    # run_demo()
    root_dir = '../../../big_data/'
    sub_dir1 = '03_fragment'
    for frag_sentence_type in ['tandem', 'parallel']:
        # frag_sentence_type = 'tandem'  # parallel or tandem
        if frag_sentence_type == 'parallel':
            sub_dir2 = '06_model_Parallel2vec'
        else:
            sub_dir2 = '05_model_Tandem2vec'
        mol_sentence_file_name = '{}_frag_smiles_sentence.csv'.format(frag_sentence_type)
        frag2num_count_file_name = 'frag2num_count.csv'
        # mol_sentence_file_name = 'step2_tandem_frag_smiles_sentence.csv'

        # calculate molecular descriptor(MD) of fragments
        frag_smiles2md_fp = os.path.join(root_dir, sub_dir1, 'frag_smiles2md.csv')
        if not os.path.exists(frag_smiles2md_fp):
            frag_id2vec = pd.read_csv(os.path.join(root_dir, sub_dir1, frag2num_count_file_name), index_col='fragment')
            frag_smiles = frag_id2vec.index.to_list()
            # frag_smiles = [i for i in frag_smiles if i != 'UNK']  # remove rare fragment token
            # print(frag_smiles)
            md = cal_md_by_smiles(frag_smiles, print_info=True)
            md.to_csv(frag_smiles2md_fp, index_label='fragment')

        mol_sent_file_replace_uncommon_frag = re.sub(r'(\..*)', r'_replaced_uncommon_frag\g<1>', mol_sentence_file_name)
        insert_unk(corpus=os.path.join(root_dir, sub_dir1, mol_sentence_file_name),
                   out_corpus=os.path.join(root_dir, sub_dir1, mol_sent_file_replace_uncommon_frag),
                   threshold=5)

        frag2count_with_unk_fp = os.path.join(root_dir, sub_dir1, 'frag2count_with_UNK.csv')
        if not os.path.exists(frag2count_with_unk_fp):
            frag2count = count_frag_in_mol_sentence(os.path.join(root_dir, sub_dir1, mol_sent_file_replace_uncommon_frag))
            frag2count.to_csv(frag2count_with_unk_fp, index_label='fragment')

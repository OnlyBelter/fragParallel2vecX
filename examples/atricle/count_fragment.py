import os
import re
import fragt2vec
import pandas as pd
from gensim.test.utils import get_tmpfile
from fragt2vec.utility import count_frag_in_mol_sentence
from fragt2vec.utility import insert_unk
import pkg_resources


def run_demo():
    demo_dir = pkg_resources.resource_filename('fragt2vec', 'demo_data')
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
    root_dir = '../../big_data/'
    # mol_sentence_file_name = 'step2_parallel_frag_smiles_sentence.csv'
    mol_sentence_file_name = 'step2_parallel_frag_smiles_sentence.csv'
    mol_sf_replace_uncommon_frag = re.sub(r'(\..*)', r'_replaced_uncommon_frag\g<1>', mol_sentence_file_name)
    sub_dir1 = '03_fragment'
    frag_info = pd.read_csv(os.path.join(root_dir, sub_dir1, 'frag_info_marked.csv'), index_col=0)
    need_replace_frag = frag_info[frag_info['keep'] == 0].index.to_list()
    insert_unk(corpus=os.path.join(root_dir, sub_dir1, mol_sentence_file_name),
               out_corpus=os.path.join(root_dir, sub_dir1, mol_sf_replace_uncommon_frag),
               threshold=19, need_replace_list=tuple(need_replace_frag))
    frag2count = count_frag_in_mol_sentence(os.path.join(root_dir, sub_dir1, mol_sf_replace_uncommon_frag))
    frag2count.to_csv(os.path.join(root_dir, sub_dir1, 'frag2count_' + mol_sentence_file_name.replace('txt', 'csv')),
                      index_label='fragment')

import os
import pandas as pd
from fragpara2vec.utility import find_bond_pair, find_aromatic_non_aroma_ring_pair

if __name__ == '__main__':
    root_dir = '../../../big_data'
    sub_dir = '05_model_Tandem2vec'
    frag_info_file_name = 'frag_smiles2md.csv'
    bond_pair_result = 'bond_pairs.txt'
    # bond_type = 'triple_bond'  # double_bond or triple_bond
    frag_info_path = os.path.join(root_dir, sub_dir, frag_info_file_name)
    result_file_path = os.path.join(root_dir, sub_dir, bond_pair_result)
    frag_info = pd.read_csv(frag_info_path, index_col=0)

    # find double_bond or triple_bond pairs
    # print(type(bond_pair))
    if not os.path.exists(result_file_path):
        with open(result_file_path, 'w') as f_handle:
            f_handle.write('\t'.join(['frag1', 'frag2', 'bond_type']) + '\n')
    for bond_type in ['double_bond', 'triple_bond']:
        bond_pair = find_bond_pair(frag_df=frag_info, bond_type=bond_type)
        with open(result_file_path, 'a') as f_handle:
            if bond_pair:
                for bp in bond_pair:
                    # print(bp)
                    f_handle.write('\t'.join(list(bp) + [bond_type]) + '\n')

    # find aromatic ring and corresponding non-aromatic ring pair
    bond_pair = find_aromatic_non_aroma_ring_pair(frag_df=frag_info)
    with open(result_file_path, 'a') as f_handle:
        if bond_pair:
            for bp in bond_pair:
                # print(bp)
                f_handle.write('\t'.join(list(bp) + ['aromatic_ring']) + '\n')

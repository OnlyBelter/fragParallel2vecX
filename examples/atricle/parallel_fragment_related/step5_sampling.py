import os
import pandas as pd
from tqdm import tqdm
from fragpara2vec.mlp import get_class_md_combination, down_sampling_mol
from fragpara2vec.utility import cal_md_by_smiles, get_ordered_md, print_df, grouper


if __name__ == '__main__':
    root_dir = r'F:\github\fragParallel2vecX\big_data'
    sub_dir1 = '02_filtered_molecule'
    sub_dir2 = '03_fragment'
    current_sub_dir = '07_sampling'
    current_dir = os.path.join(root_dir, current_sub_dir)
    cid2md_file_name = 'cid2MD.csv'    # 19 million
    cid2md_file_path = os.path.join(root_dir, current_sub_dir, cid2md_file_name)
    cid2md_file_name_new = 'cid2MD_new.csv'  # 16 million, after filtering and refragment
    cid2md_file_path_new = os.path.join(current_dir, cid2md_file_name_new)

    cid_in_mol_sentence = {}
    ordered_md = get_ordered_md()
    print('Start to calculate MD of each molecule...')
    if not os.path.exists(cid2md_file_path):
        cid2smiles_file_name = 'cid2SMILES_filtered.txt'
        cid2smile_file_path = os.path.join(root_dir, sub_dir1, cid2smiles_file_name)
        cid2sentence_file_name = 'tandem_cid2smiles_sentence.csv'  # one CID one line
        with open(cid2md_file_path, 'w') as f:
            f.write('\t'.join(['cid'] + ordered_md) + '\n')
        with open(os.path.join(root_dir, sub_dir2, cid2sentence_file_name), 'r') as file_handler:
            for line in file_handler:
                cid, _ = line.strip().split('\t')
                cid_in_mol_sentence[cid] = 1
        with open(cid2smile_file_path, 'r') as f:
            for lines in grouper(f, 200000, ''):
                # current_smiles_list = []
                current_smiles2cid = {}
                current_smiles2md = {}
                for line in lines:
                    cid, smiles = line.strip().split('\t')
                    if cid != 'cid' and cid in cid_in_mol_sentence:
                        current_smiles2cid[smiles] = cid
                current_smiles_list = list(current_smiles2cid.keys())
                smiles2md_df = cal_md_by_smiles(smiles_list=current_smiles_list)
                smiles2md_df['cid'] = smiles2md_df.index.map(current_smiles2cid)
                smiles2md_df = smiles2md_df.loc[:, ['cid'] + ordered_md]
                smiles2md_df.to_csv(cid2md_file_path, mode='a', header=False, index=False)
    else:
        print('>>> Using previous result...')

    # using previous result, after refragment, 20201018
    if not os.path.exists(cid2md_file_path_new):
        cid2md = pd.read_csv(cid2md_file_path, index_col=0)
        cid2smiles_file_name = os.path.join(root_dir, sub_dir2, 'cid2frag_info', 'cid2mol_smiles.txt')
        cid_list_new = {}
        with open(cid2smiles_file_name, 'r') as f:
            for i in tqdm(f):
                cid = i.split('\t')[0]
                if cid != 'cid':
                    cid_list_new[int(cid)] = 1
        cid2md.loc[list(cid_list_new.keys()), :].to_csv(cid2md_file_path_new, index_label='cid')

    print('Start to class molecules by combination by MD...')
    class_by_md_combination_file_path = os.path.join(current_dir, 'class_by_md_combination.csv')
    if not os.path.exists(class_by_md_combination_file_path):
        cid2md = pd.read_csv(cid2md_file_path_new, index_col='cid')
        # assert cid2md.shape[1] == 9
        print_df(cid2md)
        class_by_md_comb = get_class_md_combination(cid2md)
        print('   >check again')
        print(sum(class_by_md_comb.index.isin(cid2md.index)))
        print_df(class_by_md_comb)
        class_by_md_comb.to_csv(class_by_md_combination_file_path, index_label='cid')
    else:
        print('>>> Using previous result...')
        # class_by_md_comb = pd.read_csv(class_by_md_combination_file_path, index_col=0)
    print('Start to down-sampling...')
    selected_cid2md_class_file_path = os.path.join(current_dir, 'selected_cid2md_class.csv')
    if not os.path.exists(selected_cid2md_class_file_path):
        down_sampling_mol(class_by_md_combination_file_path, result_dir=current_dir, max_n=2000)
    # count selected class and get MD of selected mol
    cid2md = pd.read_csv(cid2md_file_path_new, index_col=0)
    selected_mol = pd.read_csv(selected_cid2md_class_file_path, index_col=0)
    cid2md_selected = cid2md.loc[selected_mol.index, :].copy()
    cid2md_selected.to_csv(os.path.join(current_dir, 'selected_cid2md.csv'), index_label='cid')
    selected_mol['n_class'] = 1
    selected_mol_class_count = selected_mol.groupby(['md_class']).count()
    selected_mol_class_count.to_csv(os.path.join(current_dir, 'n_class_selected_cid.csv'))

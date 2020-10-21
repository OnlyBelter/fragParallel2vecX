import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from fragpara2vec.utility import print_df
from fragpara2vec.utility import find_bond_pair, find_aromatic_non_aroma_ring_pair


if __name__ == '__main__':
    root_dir = '../../../big_data'
    frag_type = 'parallel'  # tandem or parallel
    if frag_type == 'tandem':
        sub_dir = '05_model_Tandem2vec'
    else:
        sub_dir = '06_model_Parallel2vec'
    sub_dir_frag = '03_fragment'
    frag_info_file_name = 'frag_smiles2md.csv'
    bond_pair_file = 'bond_pairs.txt'
    frag_info_path = os.path.join(root_dir, sub_dir_frag, frag_info_file_name)
    bond_pair_file_path = os.path.join(root_dir, sub_dir, bond_pair_file)
    frag_smiles2vec_file_path = os.path.join(root_dir, sub_dir,
                                             'frag_smiles2vec_minn_1_maxn_2_{}.csv'.format(frag_type.lower()))

    frag_smiles2vec = pd.read_csv(frag_smiles2vec_file_path, index_col=0)
    frag_info = pd.read_csv(frag_info_path, index_col=0)
    # only keep the fragments which have fragment vector
    frag_info = frag_info[frag_info.index.isin(frag_smiles2vec.index)]

    if not os.path.exists(bond_pair_file_path):
        print('Start to find bond pairs...')
        with open(bond_pair_file_path, 'w') as f_handle:
            f_handle.write('\t'.join(['frag1', 'frag2', 'bond_type']) + '\n')

        # find double_bond or triple_bond pairs
        # print(type(bond_pair))
        for bond_type in ['double_bond', 'triple_bond']:
            bond_pair = find_bond_pair(frag_df=frag_info, bond_type=bond_type)
            with open(bond_pair_file_path, 'a') as f_handle:
                if bond_pair:
                    for bp in bond_pair:
                        # print(bp)
                        f_handle.write('\t'.join(list(bp) + [bond_type]) + '\n')

        # find aromatic ring and corresponding non-aromatic ring pair
        bond_pair = find_aromatic_non_aroma_ring_pair(frag_df=frag_info)
        with open(bond_pair_file_path, 'a') as f_handle:
            if bond_pair:
                for bp in bond_pair:
                    # print(bp)
                    f_handle.write('\t'.join(list(bp) + ['aromatic_ring']) + '\n')

    frag_pairs = pd.read_csv(bond_pair_file_path, sep='\t')
    frag2vec = pd.read_csv(frag_smiles2vec_file_path, index_col='fragment')
    pca = PCA(n_components=2)
    x_reduced_pca = pd.DataFrame(data=pca.fit_transform(frag2vec), index=frag2vec.index)
    print('>>> x_reduced_pca')
    print_df(x_reduced_pca)

    # frag_pairs = frag_pairs.loc[frag_pairs['keep'] == 1]
    print('>>> frag_pairs')
    print_df(frag_pairs)

    for bond_type in frag_pairs['bond_type'].unique():
        print('>>> Deal with {}...'.format(bond_type))
        plt.figure(figsize=(8, 6))
        current_bond_pairs = frag_pairs.loc[frag_pairs['bond_type'] == bond_type].copy()
        if current_bond_pairs.shape[0] > 100:
            current_bond_pairs = current_bond_pairs.sample(n=100, random_state=42)
        current_frag1 = current_bond_pairs.loc[:, 'frag1']
        current_frag2 = current_bond_pairs.loc[:, 'frag2']
        assert current_frag1.shape == current_frag2.shape
        frag1_vec = x_reduced_pca.loc[current_frag1, :]
        frag2_vec = x_reduced_pca.loc[current_frag2, :]
        # if bond_type == 'double_bond':
        label1 = 'double bond'
        label2 = 'single bond'
        if bond_type == 'triple_bond':
            label1 = 'triple bond'
            label2 = 'single bond'
        elif bond_type == 'aromatic_ring':
            label1 = 'aromatic ring'
            label2 = 'non-aromatic ring'
        plt.scatter(frag1_vec[0], frag1_vec[1], s=6, label=label1)
        plt.scatter(frag2_vec[0], frag2_vec[1], s=6, label=label2)
        for i in range(frag1_vec.shape[0]):
            #         print(frag1_vec.loc[i,:].index)
            # plt.text(frag1_vec.iloc[i, 0], frag1_vec.iloc[i, 1] + 0.1, frag1_vec.iloc[i, :].name)
            # plt.text(frag2_vec.iloc[i, 0], frag2_vec.iloc[i, 1] - 0.1, frag2_vec.iloc[i, :].name)
            plt.plot([frag1_vec.iloc[i, 0], frag2_vec.iloc[i, 0]],
                     [frag1_vec.iloc[i, 1], frag2_vec.iloc[i, 1]],
                     c='g', alpha=0.6, linestyle='dashed')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(root_dir, sub_dir,
                                 '{}_pairs.png'.format(bond_type)), dpi=200)
        plt.close()

import os
import pandas as pd
from fragpara2vec.mlp import predict
from fragpara2vec.utility import (SELECTED_MD, get_format_time, find_nearest_neighbor,
                                  draw_multiple_mol, show_each_md, reduce_by_tsne)


if __name__ == '__main__':
    root_dir = r'F:\github\fragParallel2vecX\big_data\08_train_mol_vec_by_mlp\tree_decomposition'
    mol_vec_100d_file = 'mol_vec_tandem_frag_minn_1_maxn_2.csv'
    mol_vec_30d_file = 'mol_vec_tandem_frag_minn_1_maxn_2_new_30d.csv'
    test_set_file_path = r'F:\github\fragParallel2vecX\big_data\08_train_mol_vec_by_mlp\test_set.csv'
    mol2md_file_path = r'F:\github\fragParallel2vecX\big_data\07_sampling\selected_cid2md.csv'
    x_reduced_30d_file_path = os.path.join(root_dir, 'x_reduced_30d.csv')
    x_reduced_100d_file_path = os.path.join(root_dir, 'x_reduced_100d.csv')
    if (not os.path.exists(x_reduced_30d_file_path)) and (not os.path.exists(x_reduced_100d_file_path)):
        test_set = pd.read_csv(test_set_file_path, index_col=0)
        mol_vec_30d = pd.read_csv(os.path.join(root_dir, mol_vec_30d_file), index_col=0)
        mol_vec_100d = pd.read_csv(os.path.join(root_dir, mol_vec_100d_file), index_col=0)
        mol_vec_30d = mol_vec_30d.loc[test_set.index, :].copy()
        mol_vec_100d = mol_vec_100d.loc[test_set.index, :].copy()
        x_reduced_100d = pd.DataFrame(data=reduce_by_tsne(mol_vec_100d), index=test_set.index)
        x_reduced_30d = pd.DataFrame(data=reduce_by_tsne(mol_vec_30d), index=test_set.index)
        x_reduced_30d.to_csv(x_reduced_30d_file_path)
        x_reduced_100d.to_csv(x_reduced_100d_file_path)
    else:
        print('>>> using previous result...')
        x_reduced_30d = pd.read_csv(x_reduced_30d_file_path, index_col=0)
        x_reduced_100d = pd.read_csv(x_reduced_100d_file_path, index_col=0)
        x_reduced_30d.columns = x_reduced_30d.columns.astype(int)
        x_reduced_100d.columns = x_reduced_100d.columns.astype(int)
    mol2md = pd.read_csv(mol2md_file_path, index_col=0)
    md_list = ['nO', 'nN', 'nP', 'nS']
    for d in [30, 100]:
        save_fig_path = os.path.join(root_dir,
                                     't-SNE_vis_ws_{}_minn_{}_maxn_{}_{}_{}d.png'.format(5, 1, 2, 'tandem', d))
        if d == 30:
            x_reduced = x_reduced_30d
        else:
            x_reduced = x_reduced_100d
        fig = show_each_md(x_reduced=x_reduced, frag_info=mol2md,
                           md_list=md_list, trim=True)
        fig.savefig(save_fig_path, dpi=200)
        fig.close()

import os
import pandas as pd
import numpy as np
from tensorflow import keras
from fragpara2vec.mlp import nn_model_regression
from fragpara2vec.utility import SELECTED_MD, print_df, get_format_time, \
    find_nearest_neighbor, draw_multiple_mol, show_each_md, reduce_by_tsne


def save_fig(fig, file_path):
    with open(file_path, 'w') as f_handle:
        try:
            f_handle.write(fig)
        except TypeError:
            f_handle.write(fig.data)


if __name__ == '__main__':
    model_type = 'regression'  # classification
    root_dir = r'F:\github\fragTandem2vecX\big_data\05_model_Tandem2vec'
    result_dir = r'F:\github\fragTandem2vecX\big_data\06_train_frag_by_mlp'
    frag2md_info_file = 'frag_smiles2md.csv'
    frag2vec_file = 'frag_id2vec_minn_1_maxn_2.csv'

    # deal with y
    print('  > Start to deal with y...')
    frag2md_info = pd.read_csv(os.path.join(root_dir, frag2md_info_file), index_col='fragment')
    if model_type == 'classification':
        frag2md_info[frag2md_info >= 1] = 1
    all_x = pd.read_csv(os.path.join(root_dir, frag2vec_file), index_col=0)
    x = all_x.loc[all_x.index != 'UNK'].copy()
    y = frag2md_info.loc[x.index, SELECTED_MD].copy()

    # train model
    print('  > Start to train model...')
    for md in SELECTED_MD:
        if md == 'naRing':
            result_dir_new = os.path.join(result_dir, md + '_aromaticity')
            m_part1 = nn_model_regression(x=x, y=y.loc[:, [md]], epochs=100,
                                          result_dir=result_dir_new, callback=True)
        else:
            result_dir_new = os.path.join(result_dir, md)
            m_part1 = nn_model_regression(x=x, y=y.loc[:, [md]], epochs=100,
                                          result_dir=result_dir_new, callback=True)

        # get new frag_id2vec in 30D
        frag2vec_30d = pd.DataFrame(data=m_part1.predict(all_x), index=all_x.index)
        print_df(frag2vec_30d)
        frag2vec_new_fp = os.path.join(result_dir_new, 'frag2vec_new_30d.csv')
        frag2vec_30d.to_csv(frag2vec_new_fp)

        # plot by t-SNE
        x_reduced = reduce_by_tsne(frag2vec_30d)
        x_reduced = pd.DataFrame(data=x_reduced, index=frag2vec_30d.index)
        print('  >Start to plot t-SNE vis of fragment vector...')
        # save_fig_path = os.path.join('./chapter4_figure/', 't-SNE_vis_new_30d.png')
        need_plot_md = [md] + list(np.random.choice(SELECTED_MD, 3, replace=False))
        fig = show_each_md(x_reduced=x_reduced, frag_info=frag2md_info.loc[:, need_plot_md])
        fig.savefig(os.path.join(result_dir_new, 't-SNE_vis_sorted_by_{}.png'.format(md)), dpi=200)

        # plot top n fragment
        topn = 4
        print('  >Start to plot top {} nearest neighbor of selected fragment vector...'.format(topn))
        q_frags = ["C1=COCO1", "C1=CCNN=C1", "C1=CCC1", "OBr", "S=S", "C1#CNCC1"]
        q_mol2vec = frag2vec_30d.loc[q_frags, :]
        nn = find_nearest_neighbor(training_mol_vec_fp=frag2vec_new_fp, query_mol_vec_df=q_mol2vec, top_n=topn)
        # plot
        smiles_list = []
        dis = []
        legends = []
        for inx in range(len(q_frags)):
            smiles_list += [i.split(": ")[0] for i in nn[inx][q_frags[inx]].split('; ')]
            dis += [str('{:.8f}').format(float(i.split(": ")[1])) for i in nn[inx][q_frags[inx]].split('; ')]
            # print(dis)
            # print(inx, smiles_list)
        legends += ['{}({})'.format(smiles_list[i], dis[i]) for i in range(len(smiles_list))]
        fig = draw_multiple_mol(smiles_list=smiles_list, mols_per_row=topn, legends=legends)
        # print(type(fig))
        # print(fig)
        save_fig(fig, file_path=os.path.join(result_dir_new, 'top{}_nearest_neighbor_sorted_by_{}.svg'.format(topn, md)))

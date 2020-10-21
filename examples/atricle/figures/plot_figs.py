import os

import argparse
import pandas as pd

from fragpara2vec.utility import (SELECTED_MD, get_format_time, find_nearest_neighbor,
                                  draw_multiple_mol, show_each_md, reduce_by_tsne, print_df)

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

import rdkit
from rdkit.Chem.Draw import IPythonConsole
# IPythonConsole.ipython_useSVG = True
from IPython.display import SVG
from sklearn.decomposition import PCA

import rdkit.Chem as Chem
from rdkit.Chem import Draw


if __name__ == '__main__':
    root_dir = '../../../big_data'
    # result_dir = os.path.join(root_dir, 'figures', 'chapter3_figures')
    # sub_dir = '06_model_Parallel2vec'  # 05_model_Tandem2vec or 06_model_Parallel2vec
    # input_dir = os.path.join(root_dir, sub_dir)
    need_plot_md = ['nN', 'nS', 'nBondsD', 'naRing']
    frag2info = pd.read_csv(os.path.join(root_dir, '03_fragment', 'frag_smiles2md.csv'), index_col=0)
    frag2info = frag2info.loc[:, need_plot_md]
    print_df(frag2info)
    # if not os.path.exists(result_dir):
    #     os.makedirs(result_dir)

    for frag_sentence_type in ['parallel']:  # only plot parallel
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
        for minn, maxn in [(0, 0)]:  # [(0, 0), (1, 2)]
            print('>>> Deal with minn: ({}), maxn: ({})'.format(minn, maxn))
            frag2vec_file_name = 'frag_smiles2vec_minn_{}_maxn_{}_{}.csv'.format(minn, maxn, frag_sentence_type)
            # model_file_name = '{}2vec_model_minn_{}_maxn_{}.bin'.format(frag_sentence_type, minn, maxn)
            # model_file_name = 'tandem2vec_model.bin'
            frag_smiles2vec_file_path = os.path.join(root_dir, sub_dir2, frag2vec_file_name)

            # frag_id2vec_file = 'frag_id2vec_minn_{}_maxn_{}.csv'.format(minn, maxn)
            # frag2vec_fp = os.path.join(root_dir, frag_id2vec_file)
            x_reduced_file_path = os.path.join(root_dir, sub_dir2,
                                               'frag2vec_reduced_by_tSNE_MINN_{}_MAXN_{}.csv'.format(minn, maxn))
            if not os.path.exists(x_reduced_file_path):
                frag2vec = pd.read_csv(frag_smiles2vec_file_path, index_col='fragment')
                x_reduced = reduce_by_tsne(frag2vec)
                x_reduced = pd.DataFrame(data=x_reduced, index=frag2vec.index)
                x_reduced.to_csv(x_reduced_file_path)
            else:
                x_reduced = pd.read_csv(x_reduced_file_path, index_col=0)
                if minn == 0:
                    x_reduced = x_reduced * 10e7  # scaling tiny values
                x_reduced.columns = [int(i) for i in x_reduced.columns]
            # fig. 3-2, 3-3
            print('  >Start to plot t-SNE vis of fragment vector...')
            save_fig_path = os.path.join(root_dir, sub_dir2,
                                         't-SNE_vis_ws_{}_minn_{}_maxn_{}_{}.png'.format(4, minn, maxn,
                                                                                         frag_sentence_type))
            fig = show_each_md(x_reduced=x_reduced, frag_info=frag2info)
            fig.savefig(save_fig_path, dpi=200)
            fig.close()

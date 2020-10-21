import os
import pandas as pd
import numpy as np
from tqdm import tqdm
# from tensorflow import keras
from fragpara2vec.mlp import query_smiles_by_cid
from fragpara2vec.Parallel2vec import call_mol_tree

from fragpara2vec.mlp.nn import nn_model_regression, split_data_set, load_pre_trained_model
from fragpara2vec.utility import (SELECTED_MD, print_df, find_nearest_neighbor, cal_md_by_smiles,
                                  draw_multiple_mol, show_each_md, reduce_by_tsne,
                                  get_mol_vec, grouper, get_ordered_md)


def save_fig(fig, file_path):
    with open(file_path, 'w') as f_handle:
        try:
            f_handle.write(fig)
        except TypeError:
            f_handle.write(fig.data)


def archive():
    # deal with y
    print('  > Start to deal with y...')
    frag2md_info_file = ''
    frag2vec_file = ''
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
        nn = find_nearest_neighbor(training_mol_vec_fp=frag2vec_new_fp,
                                   query_mol_vec_df=q_mol2vec, top_n=topn, mol2md_fp='')
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
        save_fig(fig,
                 file_path=os.path.join(result_dir_new, 'top{}_nearest_neighbor_sorted_by_{}.svg'.format(topn, md)))


if __name__ == '__main__':
    model_type = 'regression'  # or classification, regression is better
    root_dir = r'F:\github\fragParallel2vecX\big_data'
    subdir_fragment = r'03_fragment/cid2frag_info'
    subdir1 = '07_sampling'
    subdir2 = '08_train_mol_vec_by_mlp'
    subdir_tandem = '05_model_Tandem2vec'
    subdir_parallel = '06_model_Parallel2vec'
    subdir_Mol2vec = '04_model_Mol2vec'
    cid2smiles_file = 'cid2mol_smiles.txt'
    result_dir = os.path.join(root_dir, subdir2)
    down_sampled_mol_file = 'selected_cid2md_class.csv'
    selected_mol2md_file_path = os.path.join(result_dir, 'selected_cid2md.csv')  # y

    # query the SMILES for down-sampled molecules by CIDs of selected molecules
    selected_cid2smiles_file_path = os.path.join(result_dir, 'selected_cid2smiles.csv')
    if not os.path.exists(selected_cid2smiles_file_path):
        print('Query the SMILES for down-sampled molecules...')
        cid2smiles_file_path = os.path.join(root_dir, subdir_fragment, cid2smiles_file)
        down_sampled_mol_file_path = os.path.join(root_dir, subdir1, down_sampled_mol_file)
        query_smiles_by_cid(cid2smiles_file_path=cid2smiles_file_path,
                            selected_mol_file_path=down_sampled_mol_file_path,
                            result_file_path=selected_cid2smiles_file_path)

    # calculate MD for selected molecules by SMILES, y
    if not os.path.exists(selected_mol2md_file_path):
        print('Calculate MD by molecular SMILES...')
        smiles2cid = {}
        with open(selected_cid2smiles_file_path, 'r', encoding='utf-8') as cid2smiles_f_handle:
            for lines in grouper(cid2smiles_f_handle, n=100000):
                smiles_list = []
                for line in lines:
                    if line is not None:
                        _cid, _smiles = line.strip().split(',')
                        if _cid != 'cid':
                            smiles2cid[_smiles] = _cid
                            smiles_list.append(_smiles)
                cid2md = cal_md_by_smiles(smiles_list=smiles_list, molecule_md=True)
                cid2md['cid'] = cid2md.index.map(smiles2cid)
                if not os.path.exists(selected_mol2md_file_path):
                    cid2md.to_csv(selected_mol2md_file_path, index_label='smiles')
                else:
                    cid2md.to_csv(selected_mol2md_file_path, header=None, mode='a')

    # get X
    # fragmentation by tree decomposition first,
    # then calculate molecular vector one by one depends on fragments and the vector of fragment
    cid2frag_tree_deco_dir = os.path.join(result_dir, 'tree_decomposition')
    if not os.path.exists(cid2frag_tree_deco_dir):
        os.makedirs(cid2frag_tree_deco_dir)
    cid2frag_smiles_file_path = os.path.join(cid2frag_tree_deco_dir, 'cid2frag_smiles.txt')
    if not os.path.exists(cid2frag_smiles_file_path):
        print('Start to generate fragments by tree decomposition...')
        call_mol_tree(raw_data_file=selected_cid2smiles_file_path,
                      result_dir=cid2frag_tree_deco_dir,
                      log_file='errors.log',
                      refragment=True,  # this should be true
                      only_fragment=True)
    for frag_type in ['tandem', 'parallel']:  # ['tandem', 'parallel']
        for minn, maxn in [(1, 2)]:
            mol_vector_file_path = os.path.join(cid2frag_tree_deco_dir,
                                                'mol_vec_{}_frag_minn_{}_maxn_{}.csv'.format(frag_type, minn, maxn))
            if not os.path.exists(mol_vector_file_path):
                print('>>> generate mol2vec by {} fragmentation...'.format(frag_type))
                if frag_type == 'tandem':
                    _current_subdir = subdir_tandem
                else:
                    _current_subdir = subdir_parallel
                frag_vec_file_path = os.path.join(root_dir, _current_subdir,
                                                  'frag_smiles2vec_minn_{}_maxn_{}_{}.csv'.format(minn, maxn, frag_type))
                print(frag_vec_file_path)
                get_mol_vec(frag2vec_file_path=frag_vec_file_path,
                            data_set=cid2frag_smiles_file_path,
                            result_path=mol_vector_file_path)

    train_set_file_path = os.path.join(result_dir, 'train_set.csv')
    test_set_file_path = os.path.join(result_dir, 'test_set.csv')
    if not (os.path.exists(train_set_file_path) and os.path.exists(test_set_file_path)):
        print('Split down-sampled dataset...')
        split_data = split_data_set(os.path.join(root_dir, subdir1, down_sampled_mol_file))
        train_set = split_data['train_set']
        test_set = split_data['test_set']
        print('>>> Training set')
        print_df(train_set)
        print('>>> Test set')
        print_df(test_set)
        train_set.to_csv(train_set_file_path)
        test_set.to_csv(test_set_file_path)

    # --------------------------------------------------------------------------------------
    # train model
    print('Start to train MLP model...')
    train_set = pd.read_csv(train_set_file_path, index_col='cid')
    selected_mol2md = pd.read_csv(selected_mol2md_file_path, index_col='cid')
    md = get_ordered_md()
    selected_mol2md = selected_mol2md.loc[:, md].copy()
    y = selected_mol2md.loc[selected_mol2md.index.isin(train_set.index)]
    for frag_type in ['tandem', 'parallel', 'random']:  # ['tandem', 'parallel']
        # for minn, maxn in [(1, 3)]:
        mlp_model_result_dir = os.path.join(result_dir, 'pre_trained_model', frag_type)
        if not os.path.exists(os.path.join(mlp_model_result_dir, 'model_reg_{}.h5'.format(frag_type))):
            print('>>> training model of {}'.format(frag_type))
            # mlp_model_result_dir = os.path.join(result_dir, 'pre_trained_model', frag_type)
            if not os.path.exists(mlp_model_result_dir):
                os.makedirs(mlp_model_result_dir)
            if frag_type != 'random':
                mol_vec_file_path = os.path.join(cid2frag_tree_deco_dir,
                                                 'mol_vec_{}_frag_minn_{}_maxn_{}.csv'.format(frag_type, 1, 2))
                mol_vec = pd.read_csv(mol_vec_file_path, index_col=0, header=None)
                x = mol_vec.loc[mol_vec.index.isin(train_set.index), :]
            else:
                random_mol2vec_file_path = os.path.join(result_dir, frag_type, 'mol_vec_random_frag.csv')
                if not os.path.exists(random_mol2vec_file_path):
                    random_mol2vec = pd.DataFrame(data=np.random.random((selected_mol2md.shape[0], 100)),
                                                  index=selected_mol2md.index)
                    random_mol2vec.to_csv(random_mol2vec_file_path, header=False)
                else:
                    random_mol2vec = pd.read_csv(random_mol2vec_file_path, index_col=0)
                x = random_mol2vec.loc[train_set.index, :]
            nn_model_regression(x=x, y=y, epochs=1000, callback=True, learning_rate=0.0002,
                                result_dir=mlp_model_result_dir, frag_type=frag_type)

    # predict new molecular vectors
    frag_type = 'tandem'
    new_mol_vec_file_path = os.path.join(cid2frag_tree_deco_dir,
                                         'mol_vec_{}_frag_minn_{}_maxn_{}_new_30d.csv'.format(frag_type, 1, 2))
    if not os.path.exists(new_mol_vec_file_path):
        print('start to get new mol vector...')
        mol_vec_file_path = os.path.join(cid2frag_tree_deco_dir,
                                         'mol_vec_{}_frag_minn_{}_maxn_{}.csv'.format(frag_type, 1, 2))
        mol_vec = pd.read_csv(mol_vec_file_path, index_col=0, header=None)
        model_part1_path = os.path.join(result_dir, 'pre_trained_model', frag_type, 'm_part1_reg_{}.h5'.format(frag_type))
        model_part1 = load_pre_trained_model(model_part1_path)
        new_mol_vec = model_part1.predict(mol_vec)

        pd.DataFrame(data=new_mol_vec, index=mol_vec.index).to_csv(new_mol_vec_file_path, index_label='cid')

    # for frag_type in ['tandem', 'parallel']:
    #     print('>>> training model of {}'.format(frag_type))
    #     mlp_model_result_dir = os.path.join(result_dir, 'pre_trained_model', frag_type)
    #     _model = None

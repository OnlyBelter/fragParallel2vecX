import os
import pandas as pd
from fragpara2vec.utility import query_smiles_by_cids
from fragpara2vec.utility import find_nearest_neighbor, draw_multiple_mol


def save_fig(fig, file_path):
    with open(file_path, 'w') as f_handle:
        try:
            f_handle.write(fig)
        except TypeError:
            f_handle.write(fig.data)


if __name__ == '__main__':
    root_dir = r'F:\github\fragParallel2vecX\big_data'
    sub_dir2 = '08_train_mol_vec_by_mlp'
    q_mol_dir = r'F:\github\fragParallel2vecX\figures\chapter5'
    frag_sentence_type = 'tandem'
    using_cid = True
    q_mol = pd.read_csv(os.path.join(q_mol_dir, 'selected_10_molecules.csv'), index_col=0)

    q_frags = q_mol.loc[:, 'SMILES'].to_list()  # CID list
    # sub_dir1 = '03_fragment'
    mol2md_file_path = os.path.join(root_dir, sub_dir2, 'selected_cid2md_only_cid.csv')
    # [i in frag2vec.index for i in q_frags]
    # new mol2vec 30d
    mol2vec_file_name = 'mol_vec_tandem_frag_minn_1_maxn_2_new_30d.csv'
    # model_file_name = '{}2vec_model_minn_{}_maxn_{}.bin'.format(frag_sentence_type, minn, maxn)
    # model_file_name = 'tandem2vec_model.bin'
    mol_smiles2vec_file_path = os.path.join(root_dir, sub_dir2, 'tree_decomposition', mol2vec_file_name)
    # read the vector of fragments
    mol2vec = pd.read_csv(mol_smiles2vec_file_path, index_col=0)
    q_frag2vec = mol2vec.loc[q_mol.index, :].copy()
    # q_frag2vec = q_frag2vec.merge(q_mol.loc[:, ['SMILES']], left_index=True, right_index=True)
    # q_frag2vec.set_index('SMILES', inplace=True)
    topn = 6
    query_result = find_nearest_neighbor(training_mol_vec_fp=mol_smiles2vec_file_path,
                                         query_mol_vec_df=q_frag2vec,
                                         top_n=topn,
                                         mol2md_fp=mol2md_file_path,
                                         min_query_count=100,
                                         using_cid=using_cid)
    nn = query_result['query2nn']
    nn_smiles = []
    smiles_list = []
    cid_list = []
    dis = []
    legends = []
    if using_cid:
        for v in nn:
            for _query, _result in v.items():
                cid2distance = {i.split(': ')[0]: i.split(': ')[1] for i in _result.split('; ')}
                dis += [str('{:.6f}').format(float(i.split(': ')[1])) for i in _result.split('; ')]
                cids = list(cid2distance.keys())
                cid_list += cids
                cid2smiles = query_smiles_by_cids(cids)
                smiles_list += list(cid2smiles.values())
                # smiles2distance = [cid2smiles[i] + ': ' + j for i, j in cid2distance.items()]
                # nn_smiles[_query] = '; '.join(smiles2distance)
    # print('>>> nn: {}'.format(nn)

    else:
        for inx in range(len(q_frags)):
            smiles_list += [i.split(": ")[0] for i in nn[inx][q_frags[inx]].split('; ')]
            dis += [str('{:.6f}').format(float(i.split(": ")[1])) for i in nn[inx][q_frags[inx]].split('; ')
                    if i.split(": ")[1] != 1]
        # print(dis)
        # print(inx, smiles_list)
    if cid_list:
        legends += ['{} ({})'.format(cid_list[i], dis[i]) for i in range(len(cid_list))]
    else:
        legends += ['{} ({})'.format(smiles_list[i], dis[i]) for i in range(len(smiles_list))]
    fig = draw_multiple_mol(smiles_list=smiles_list, mols_per_row=topn, legends=legends)
    save_fig(fig, file_path=os.path.join(q_mol_dir,
                                         'top{}_minn_{}_maxn_{}_{}.svg'.format(topn, 1, 2,
                                                                               frag_sentence_type)))
    match_pattern_file_path = os.path.join(q_mol_dir,
                                           'top{}_minn_{}_maxn_{}_{}.txt'.format(topn, 1, 2,
                                                                                 frag_sentence_type))
    with open(match_pattern_file_path, 'w') as file_handle:
        file_handle.write('\t'.join(['cid', 'pattern']) + '\n')
        for cid, pattern in query_result['match_pattern'].items():
            file_handle.write('\t'.join([str(cid), pattern]) + '\n')

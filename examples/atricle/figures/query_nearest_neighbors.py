import os
import pandas as pd
from fragpara2vec.utility import find_nearest_neighbor, draw_multiple_mol


def save_fig(fig, file_path):
    with open(file_path, 'w') as f_handle:
        try:
            f_handle.write(fig)
        except TypeError:
            f_handle.write(fig.data)


if __name__ == '__main__':
    root_dir = '../../../big_data/'
    q_frags = ["C1=COCO1", "C1=CCNN=C1", "C1=CCC1", "OBr", "S=S", "C1#CNCC1"]
    sub_dir1 = '03_fragment'
    frag2md_file_path = os.path.join(root_dir, sub_dir1, 'frag_smiles2md.csv')
    # [i in frag2vec.index for i in q_frags]
    # frag2vec

    for frag_sentence_type in ['tandem', 'parallel']:
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
        for minn, maxn in [(0, 0), (1, 2)]:
            print('>>> Deal with minn: ({}), maxn: ({})'.format(minn, maxn))
            frag2vec_file_name = 'frag_smiles2vec_minn_{}_maxn_{}_{}.csv'.format(minn, maxn, frag_sentence_type)
            model_file_name = '{}2vec_model_minn_{}_maxn_{}.bin'.format(frag_sentence_type, minn, maxn)
            # model_file_name = 'tandem2vec_model.bin'
            frag_smiles2vec_file_path = os.path.join(root_dir, sub_dir2, frag2vec_file_name)

            # read the vector of fragments
            frag2vec = pd.read_csv(frag_smiles2vec_file_path, index_col=0)
            q_frag2vec = frag2vec.loc[q_frags, :].copy()
            topn = 4
            query_result = find_nearest_neighbor(training_mol_vec_fp=frag_smiles2vec_file_path,
                                                 query_mol_vec_df=q_frag2vec,
                                                 top_n=topn,
                                                 mol2md_fp=frag2md_file_path,
                                                 min_query_count=100)
            nn = query_result['query2nn']
            # print('>>> nn: {}'.format(nn))

            smiles_list = []
            dis = []
            legends = []
            for inx in range(len(q_frags)):
                smiles_list += [i.split(": ")[0] for i in nn[inx][q_frags[inx]].split('; ')]
                dis += [str('{:.6f}').format(float(i.split(": ")[1])) for i in nn[inx][q_frags[inx]].split('; ')
                        if i.split(": ")[1] != 1]
                # print(dis)
                # print(inx, smiles_list)
            legends += ['{} ({})'.format(smiles_list[i], dis[i]) for i in range(len(smiles_list))]
            fig = draw_multiple_mol(smiles_list=smiles_list, mols_per_row=topn, legends=legends)
            save_fig(fig, file_path=os.path.join(root_dir, sub_dir2,
                                                 'top{}_minn_{}_maxn_{}_{}.svg'.format(topn, minn, maxn,
                                                                                       frag_sentence_type)))
            match_pattern_file_path = os.path.join(root_dir, sub_dir2,
                                                   'top{}_minn_{}_maxn_{}_{}.txt'.format(topn, minn, maxn,
                                                                                         frag_sentence_type))
            with open(match_pattern_file_path, 'w') as file_handle:
                file_handle.write('\t'.join(['cid', 'pattern']) + '\n')
                for cid, pattern in query_result['match_pattern'].items():
                    file_handle.write('\t'.join([cid, pattern]) + '\n')

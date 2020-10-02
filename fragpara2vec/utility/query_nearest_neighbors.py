import numpy as np
import pandas as pd


def cosine_dis(v1, v2):
    """
    cosine distance between two vectors
    :param v1:
    :param v2:
    :return:
    """
    return np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))


def cosine_dis2(df, vec):
    dot_product = np.dot(df, vec)
    norm_product = np.linalg.norm(df, axis=1) * np.linalg.norm(vec)
    return np.divide(dot_product, norm_product)


def find_nearest_neighbor(training_mol_vec_fp, query_mol_vec_df, top_n, query_amount=100000):
    """
    find top_n nearest neighbors in all training set (more than 10,000,000 molecules)
    :param query_amount:
    :param training_mol_vec_fp: molecular vector of all training set
    :param query_mol_vec_df: a data frame of molecular vector as query item, index is cid
    :param top_n: top n nearest neighbors, max is 100
    :return:
    """
    # cid2dis_top = {}
    # cid2distance = {}
    query2cid_dis = {}
    query2cid_dis_top = {}
    query2nn = []
    query_len = query_mol_vec_df.shape[0]
    index2cid = {i: query_mol_vec_df.index[i] for i in range(query_len)}
    if top_n > 100:
        top_n = 100
    with open(training_mol_vec_fp, 'r') as handel:
        counter = 0
        for i in handel:
            current_line = i.split(',')
            cid = current_line[0]
            mol_vec = [float(v) for v in current_line[1:]]
            _cosine_dis = cosine_dis2(query_mol_vec_df, mol_vec)
            for j in range(query_len):
                q_cid = index2cid[j]
                # q_mol_vec = query_mol_vec_df.loc[q_cid, :]
                if q_cid not in query2cid_dis:
                    query2cid_dis[q_cid] = {}
                query2cid_dis[q_cid][cid] = _cosine_dis[j]
            # cid2distance[cid] = cosine_dis(q_mol_vec, mol_vec)
            if len(query2cid_dis[index2cid[0]]) >= 1000:
                for q_cid2 in query_mol_vec_df.index:
                    cid2distance_sorted = sorted(query2cid_dis[q_cid2].items(), key=lambda x: x[1], reverse=True)
                    # cid2distance_df = pd.DataFrame.from_dict(query2cid_dis[q_cid2], orient='index')
                    # cid2distance_df.sort_values(by=[0], inplace=True, ascending=False)
                    cid2distance_topn = cid2distance_sorted[:top_n].copy()
                    if q_cid2 not in query2cid_dis_top:
                        query2cid_dis_top[q_cid2] = {}
                    query2cid_dis_top[q_cid2].update({i[0]: i[1] for i in cid2distance_topn})
                    query2cid_dis[q_cid2] = {}
            if counter % 10000 == 0:
                print('current line: {}'.format(counter))
            if counter >= query_amount:
                break
            counter += 1
    for q_cid in query_mol_vec_df.index:
        cid2distance_sorted = sorted(query2cid_dis[q_cid2].items(), key=lambda x: x[1], reverse=True)
        cid2distance_topn = cid2distance_sorted[:top_n].copy()
        query2cid_dis_top[q_cid].update({i[0]: i[1] for i in cid2distance_topn})
        top_dis_df = pd.DataFrame.from_dict(query2cid_dis_top[q_cid], orient='index')
        top_dis_df.sort_values(by=[0], inplace=True, ascending=False)
        top_n_dis = top_dis_df.iloc[range(top_n), :].copy().to_dict()[0]
        sorted_top_n_dis = sorted(top_n_dis.items(), key=lambda x: x[1], reverse=True)
        query2nn.append({q_cid: '; '.join(i[0] + ': ' + str(i[1]) for i in sorted_top_n_dis)})
    return query2nn

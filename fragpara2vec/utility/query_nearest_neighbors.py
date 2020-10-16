import numpy as np
import pandas as pd
from .pub_func import cal_md_by_smiles, get_ordered_md
# from fragpara2vec



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


def _find_single_nn(training_mol_vec_fp, query_mol_vec, mol2md_fp, top_n, min_query_count):
    """
    algorithm 3 in my thesis, query similar fragments
    :param training_mol_vec_fp:
    :param query_mol_vec: a dict, {smiles: vec}
    :param mol2md_fp: the file path of all molecular MD
    :param top_n: max n is 100
    :param min_query_count: the minimal number of molecules contained in query cluster, min min_query_count is top_n
    :return:
    """
    # order2md = {i: md for md, i in MD_IMPORTANCE.items()}
    ordered_md = get_ordered_md()
    q_mol_smiles = list(query_mol_vec.keys())
    query_mol_md = cal_md_by_smiles(q_mol_smiles).loc[q_mol_smiles[0], ordered_md].values  # M'
    query_vec = np.array(list(query_mol_vec.values()))[0]  # f in my paper
    query_mol_md[query_mol_md > 1] = 1  # M''
    if top_n > 100:
        top_n = 100
    if min_query_count < top_n:
        min_query_count = top_n
    while 1:
        query_mol_md = [i for i in query_mol_md if i != -1]
        mols_with_same_md = {}  # M_f in my paper
        with open(mol2md_fp, 'r') as file_handle:
            md_name = []
            for line in file_handle:
                line = line.strip().split(',')
                if (line[0] == 'fragment') or (line[0] == 'smiles'):  # first line, column names
                    md_name += line[1:]
                    # print(md_name)
                else:
                    cid = line[0]
                    current_md = [int(i) if int(i) == 0 else 1 for i in line[1:]]
                    md2val = dict(zip(md_name, current_md))
                    # print(md2val)
                    current_md_val = [md2val[_md] for _md in ordered_md]
                    if np.all([query_mol_md[i] == current_md_val[i] for i in range(len(query_mol_md))]):
                        mols_with_same_md[cid] = 1
        if len(mols_with_same_md) < min_query_count:
            if -1 not in query_mol_md:
                query_mol_md[-1] = -1
            else:
                query_mol_md[query_mol_md.index(-1)-1] = -1
        else:
            break
    mol2dis = {}
    with open(training_mol_vec_fp, 'r') as file_handle:
        for line in file_handle:
            line = line.strip().split(',')
            cid = line[0]
            if cid in mols_with_same_md:
                current_vec = np.array([float(i) for i in line[1:]])
                mol2dis[cid] = cosine_dis(query_vec, current_vec)
    cid2distance_sorted = sorted(mol2dis.items(), key=lambda x: x[1], reverse=True)
    cid2distance_topn = cid2distance_sorted[:top_n]
    return {'cid2distance_topn': cid2distance_topn, 'match_pattern': ''.join([str(i) for i in query_mol_md])}


def find_nearest_neighbor(training_mol_vec_fp, query_mol_vec_df, mol2md_fp, top_n, min_query_count=1000):
    """
    find top_n nearest neighbors in all training set (more than 10,000,000 molecules)
    :param min_query_count: the minimal number of molecules contained in query cluster, min min_query_count is top_n
    :param training_mol_vec_fp: molecular vector of all training set
    :param query_mol_vec_df: a data frame of molecular vector as query item, index is cid
    :param mol2md_fp: file path of molecular descriptors
    :param top_n: top n nearest neighbors, max is 100
    :return:
    """
    query2nn = []
    query_cid2topn = {}
    query_cid2match_pattern = {}
    for cid in query_mol_vec_df.index:
        current_query_mol_vec = {cid: query_mol_vec_df.loc[cid].values}
        query_result = _find_single_nn(training_mol_vec_fp=training_mol_vec_fp,
                                       query_mol_vec=current_query_mol_vec,
                                       mol2md_fp=mol2md_fp,
                                       top_n=top_n,
                                       min_query_count=min_query_count)
        query_cid2topn[cid] = query_result['cid2distance_topn']
        query_cid2match_pattern[cid] = query_result['match_pattern']
    for q_cid, sorted_top_n_dis in query_cid2topn.items():
        query2nn.append({q_cid: '; '.join(i[0] + ': ' + str(i[1]) for i in sorted_top_n_dis)})
    return {'query2nn': query2nn, 'match_pattern': query_cid2match_pattern}

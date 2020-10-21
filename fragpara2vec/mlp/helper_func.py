import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from fragpara2vec.utility import get_ordered_md
from fragpara2vec.Parallel2vec import read_json_line


ELEMENTS = ['S', 'Br', 'O', 'C', 'F', 'P', 'N', 'I', 'Cl', 'H']
BONDS = ['DOUBLE', 'SINGLE', 'TRIPLE']
# SELECTED_MD = ['nN', 'nS', 'nO', 'nX', 'nBondsD', 'nBondsT', 'naRing', 'nARing']
PRMIER_NUM = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


def get_class_md_combination(frag_info, selected_md=None, min_number=3):
    """
    get unique class depends on different molecular descriptors
    frag_info: a dataframe
        contains fragment SMILES (or molecular CID) and all of the selected MD
    selected_md: selected molecular descriptors (MD)
    min_number: the minimal number of fragment in each class
    :return: fragment SMILES or CID, class (the combination of different MD, such as 10001010),
             class_id(0 to n), class_num(count each class)
    """
    if not selected_md:
        selected_md = get_ordered_md()
    frag_info = frag_info.loc[:, selected_md].copy()
    frag_info[frag_info >= 1] = 1
    frag2class = pd.DataFrame(columns=['md_class'], index=frag_info.index)
    frag2class['md_class'] = frag_info.apply(lambda x: ''.join([str(i) for i in x]), axis=1)

    frag_class2num = {}
    for c in tqdm(frag2class.index):
        md_class = frag2class.loc[c, 'md_class']
        if md_class not in frag_class2num:
            frag_class2num[md_class] = 0
        frag_class2num[md_class] += 1
    frag_class2num_df = pd.DataFrame.from_dict(frag_class2num, orient='index', columns=['class_num'])
    frag2class = frag2class.merge(frag_class2num_df, left_on='md_class', right_index=True)
    frag2class = frag2class[frag2class['class_num'] >= min_number].copy()
    print('  >the shape of frag2class after filtered: {}'.format(frag2class.shape))

    unique_class = sorted(frag2class['md_class'].unique())
    code2id = {unique_class[i]: i for i in range(len(unique_class))}
    print(code2id)
    frag2class['class_id'] = frag2class['md_class'].apply(lambda x: code2id[x])

    # depth = len(code2id)
    # y_one_hot = tf.one_hot(frag2class_filtered.class_id.values, depth=depth)
    # print('  >the shape of one hot y: {}'.format(y_one_hot.shape))
    print('   >>> check index in frag2class')
    print(np.all(frag2class.index.isin(frag_info.index)))
    return frag2class


def down_sampling_mol(md_comb_file_path, max_n, result_dir=None):
    """
    md_class means a binary string of 9 different MD, such as 100110100,
    which can classify all molecules in to different groups.
    Since the imbalance of each group, we need down-sampling here.
    :param md_comb_file_path: a file path of the output of function get_class_md_combination
        which contains cid,class,class_num,class_id
    :param max_n: max cids sampled from each md_class
    :param result_dir:
    :return:
    """
    class_id2info = {}
    with open(md_comb_file_path, 'r') as f:
        for _line in tqdm(f):
            cid, md_class, class_num, class_id = _line.strip().split(',')
            if cid != 'cid':
                if class_id not in class_id2info:
                    class_id2info[class_id] = {'md_class': '', 'class_num': 0, 'cids': {}}
                class_id2info[class_id]['md_class'] = md_class
                class_id2info[class_id]['class_num'] = int(class_num)
                class_id2info[class_id]['cids'][cid] = 1
    class_num = {}
    class2sampled_cid = {}
    for class_id, class_info in class_id2info.items():
        if class_id not in class_num:
            class_num[class_id] = []
        class_num[class_id] += [class_info['md_class'], class_info['class_num']]
        current_cids = list(class_info['cids'].keys())
        if len(current_cids) <= max_n:
            class2sampled_cid[class_id] = current_cids
        else:
            selected_cids = np.random.choice(current_cids, size=max_n, replace=False)
            class2sampled_cid[class_id] = list(selected_cids)
    class_num_df = pd.DataFrame.from_dict(data=class_num, orient='index', columns=['md_class', 'class_num'])
    selected_cid2md_class = {}
    for class_id, cids in class2sampled_cid.items():
        md_class = class_id2info[class_id]['md_class']
        for cid in cids:
            selected_cid2md_class[cid] = md_class
    selected_cid2md_class_df = pd.DataFrame.from_dict(data=selected_cid2md_class, orient='index', columns=['md_class'])
    if result_dir:
        class_num_df.to_csv(os.path.join(result_dir, 'num_class.csv'), index_label='class_id')
        selected_cid2md_class_df.to_csv(os.path.join(result_dir, 'selected_cid2md_class.csv'), index_label='cid')
    else:
        return {'class_num': class_num_df, 'selected_cid2md': selected_cid2md_class_df}


def query_smiles_by_cid(cid2smiles_file_path, selected_mol_file_path, result_file_path):
    """
    query the SMILES for down-sampled molecules
    :param cid2smiles_file_path: file path
        each line contains cid and smiles, the result of step1, 'cid2mol_smiles.txt'
    :param selected_mol_file_path: down-sampled result at step5 which contains cid and md_class
    :param result_file_path:
    :return:
    """

    down_sampled_mol = pd.read_csv(selected_mol_file_path, index_col='cid')
    cid_checker = {i: 1 for i in down_sampled_mol.index}
    sampled_cid2smiles = {}
    # no_smiles_cid = []
    with open(cid2smiles_file_path, 'r') as f:
        for line in tqdm(f):
            if not line.startswith('cid'):
                cid, smiles = read_json_line(line)
                cid = int(cid)
                if cid in cid_checker:
                    sampled_cid2smiles[cid] = smiles
    down_sampled_mol['smiles'] = down_sampled_mol.index.map(sampled_cid2smiles)
    down_sampled_mol_without_smiles = down_sampled_mol[down_sampled_mol['smiles'].isnull()]
    no_smiles_cid = down_sampled_mol_without_smiles.index.to_list()
    print('>>> There are {} cids ({}) haven\'t queried SMILES...'.format(len(no_smiles_cid),
                                                                         ','.join([str(i) for i in no_smiles_cid])))

    down_sampled_mol.to_csv(result_file_path, columns=['smiles'], index_label='cid')

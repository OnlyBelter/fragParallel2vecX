"""
step2: generate fragment sentence
# usage:
$ python s2_frag_sent.py ./demo_data/step1_result.txt ./demo_data --log_fn ./demo_data/step2_log.log
# plot molecular structure, molecular tree and molecular with index of the first 10 lines under test model
$ python s2_frag_sent.py ./demo_data/step1_result.txt ./demo_data --log_fn ./demo_data/step2_log.log --test

# refragment (optional)
# python s2_frag_sent.py big-data/moses_dataset/result/step1_result.txt big-data/moses_dataset/result/
# --log_fn big-data/moses_dataset/result/step2_log.log --arrangement_mode parallel
"""
import os
import json
import argparse
import datetime
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
from .pub_func import write_list_by_json
from .helper_func import Mol2Network, get_fragment_sentence, count_fragment
from ._fragment_assistant import get_clique_smiles, get_mol
from ...utility import grouper


def draw_graph(g, file_dir, file_name):
    """
    draw molecular graph
    :param g: molecular graph
    :param file_dir: where to save figure
    :param file_name: file name
    :return:
    """
    nx.draw(g, with_labels=True, font_weight='bold')
    plt.savefig(os.path.join(file_dir, file_name + '.png'), dpi=300)
    plt.close()


def mol2network(n2n, file_dir='.', file_name='_', draw_network=False):
    """

    :param n2n: node to neighbors
    :param file_dir:
    :param file_name:
    :param draw_network:
    :return: a network
    """
    g = nx.Graph()
    for i in n2n.keys():
        edges = [(i, j) for j in n2n[i]]
        g.add_edges_from(edges)
    if draw_network:
        draw_graph(g, file_dir, file_name)
    return g


def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol


def basic_test():
    SMILES = 'C#CCN(CC#C)C(=O)c1cc2ccccc2cc1OC(F)F'
    mol = Chem.MolFromSmiles(SMILES)
    id2smiles = {1: 'C#C', 2: 'CC', 3: 'CN', 4: 'CN', 5: 'CC', 6: 'C#C', 7: 'CN',
                 8: 'C=O', 9: 'CC', 10: 'CO', 11: 'CO', 12: 'CF', 13: 'CF',
                 14: 'C1=CCCC=C1', 15: 'C1=CC=CC=C1', 16: 'N', 17: 'C', 18: 'C'}
    n2n = {1: [2], 2: [1, 3], 3: [2, 16], 4: [5, 16], 5: [4, 6], 6: [5],
           7: [16, 17], 8: [17], 9: [14, 17], 10: [11, 14], 11: [10, 18], 12: [18],
           13: [18], 14: [9, 10, 15], 15: [14], 16: [3, 4, 7], 17: [7, 8, 9], 18: [11, 12, 13]}
    id2mol_inx = {"1": [0, 1], "2": [1, 2], "3": [2, 3], "4": [3, 4], "5": [4, 5], "6": [5, 6], "7": [3, 7],
                  "8": [7, 8], "9": [7, 9], "10": [18, 19], "11": [19, 20], "12": [20, 21], "13": [20, 22],
                  "14": [9, 18, 17, 16, 11, 10], "15": [12, 13, 14, 15, 16, 11], "16": [3], "17": [7], "18": [20]}
    # frag_id2mol_inx = {}
    g = mol2network(n2n)
    # mol_path = get_mol_path(n2n, g)

    print('>>> SMILES: ', SMILES)
    print('    n2n: ', n2n)
    draw_graph(g, file_dir='', file_name='test')


def writh_log_message(mes, path):
    with open(path, 'a') as f:
        f.write(mes + '\n')


def call_frag2sent(input_file_dir, arrangement_mode='parallel',
                   result_dir=None, test=False, n_line=50000):
    """
    get fragment sentence after fragmentation by tree decomposition
    :param input_file_dir: file dir of molecular fragments information from the first step,
        which contains four files:
         cid2mol_smiles.txt:  cid/smiles
         cid2frag_id2frag_smiles.txt: cid/frag_id2smiles
         cid2frag_id2neighbors: cid/frag_id2neighbors
         cid2frag_id2mol_inx.txt: cid/smiles/frag_id2mol_inx
    :param arrangement_mode: how to arrange fragments in different molecular paths of a single molecule,
        tandem/parallel (parallel is better)
    :param result_dir: result directory
    :param test: Run the entire script on only the first 10 lines and plot
    :param n_line: read n_line each time
    :return:
    """
    frag_info_files = ['cid2mol_smiles.txt', 'cid2edges.txt', 'cid2frag_id2mol_inx.txt']
    testLines = 10
    # min_len = 5
    result_file_prefix = 'tandem'
    # frag2num = pd.DataFrame()
    if arrangement_mode == 'parallel':
        result_file_prefix = 'parallel'

    print('Current arrangement mode is: {}'.format(arrangement_mode))
    result_file_cid2frag = os.path.join(result_dir, '{}_cid2smiles_sentence.csv'.format(result_file_prefix))
    result_file_frag2num = os.path.join(result_dir, '{}_frag2num_count.csv'.format(result_file_prefix))
    # frag_id or frag_smiles
    result_file_frag_smiles_sentence = os.path.join(result_dir, '{}_frag_smiles_sentence.csv'.format(result_file_prefix))

    test_fig_dir = ''
    if test:
        test_fig_dir = os.path.join(result_dir, 'test', 'figure')
        if not os.path.exists(test_fig_dir):
            os.makedirs(test_fig_dir)

    print('Start to generate fragment sentence by molecular tree...')
    print('>>> Count previous result...')
    existed_cid = {}
    if os.path.exists(result_file_cid2frag):
        with open(result_file_cid2frag, 'r', encoding='utf-8') as f:
            for line in f:
                cid, _ = line.split('\t')
                existed_cid[int(cid)] = 1
        print('>>> Fragment sentences of {} molecules have been generated'.format(len(existed_cid)))
    counter = 0
    file2n_lines = {}
    file_handles = {filename: open(os.path.join(input_file_dir, filename), 'r', encoding='utf-8')
                    for filename in frag_info_files}
    for filename, file_handle in file_handles.items():
        filename = filename.replace('.txt', '')
        file2n_lines[filename] = grouper(file_handle, n=n_line)

    while 1:
        # https://stackoverflow.com/a/46392753/2803344
        if test and counter > testLines:
            break
        cid2info = {}
        lines = None
        f_cid2frag_sentence = open(result_file_cid2frag, 'a', encoding='utf-8')
        for filename, n_line in file2n_lines.items():
            # frag_info_name = filename.replace('.txt', '')
            lines = next(n_line, None)
            for line in lines:
                if line is not None:
                    line = line.strip().split('\t')
                    if line[0] != 'cid':
                        _cid = line[0]  # not json
                        frag_info = json.loads(line[1])
                        if int(_cid) not in existed_cid:
                            if _cid not in cid2info:
                                cid2info[_cid] = {}
                            cid2info[_cid][filename] = frag_info
        if len(cid2info) != 0:
            for cid, current_row in tqdm(cid2info.items()):
                if len(current_row) == len(frag_info_files):
                    mol_smiles = current_row['cid2mol_smiles']  # molecular SMILES
                    edges = current_row['cid2edges']  # the relation between fragments
                    # atoms which contained this fragment
                    frag_id2mol_inx = {int(i): j for i, j in current_row['cid2frag_id2mol_inx'].items()}
                    if counter % 100000 == 0 or test:
                        # print('>>> current line: ', counter)
                        print('>>>CID: {}, SMILES: {}, (line {})'.format(cid, mol_smiles, counter))
                        print('>>>Current time: {}'.format(datetime.datetime.now()))
                    network = Mol2Network(smiles=mol_smiles, edges=edges, frag_id2mol_inx=frag_id2mol_inx)
                    mol_paths = network.get_mol_path()  # all mol paths, a list of lists
                    if test:
                        print('>>> current line: ', counter)
                        print('    The longest mol path of this molecule: ', mol_paths[0])
                        print('    id2smiles: ', network.id2smiles)
                        print('    n2n: ', network.n2n)

                        draw_graph(network.g, file_dir=test_fig_dir,
                                   file_name='mol_tree_cid:{}_line:{}'.format(cid, counter))
                        mol = Chem.MolFromSmiles(mol_smiles)
                        Draw.MolToFile(mol, os.path.join('big-data', 'figure',
                                                         'mol_structure_{}.png'.format(counter)))
                        mol_with_inx = mol_with_atom_index(mol)
                        Draw.MolToFile(mol_with_inx,
                                       os.path.join('big-data', 'figure',
                                                    'mol_with_inx_{}.png'.format(counter)))
                    # except Exception as e:
                    #     mol_paths = []
                    #     with open(log_file, 'a') as log_f:
                    #         log_f.write('getting mol path from network has error, cid: {}'.format(cid) + '\n')
                    if mol_paths:
                        mol_sentence = []
                        # mol_path_str = ','.join([str(i) for i in mol_paths])
                        if arrangement_mode == 'tandem':
                            for mol_path in mol_paths:  # molecular sentence with fragment SMILES
                                mol_sentence += [network.id2smiles[frag_id] for frag_id in mol_path]
                            frag_smiles = ','.join(mol_sentence)
                            f_cid2frag_sentence.write('\t'.join([cid, frag_smiles]) + '\n')
                        elif arrangement_mode == 'parallel':
                            for mol_path in mol_paths:
                                # may have multiple sentences for each molecule
                                one_frag_smiles = '\t'.join([cid, ','.join([network.id2smiles[frag_id]
                                                                            for frag_id in mol_path])])
                                mol_sentence.append(one_frag_smiles)
                            f_cid2frag_sentence.write('\n'.join(mol_sentence) + '\n')
                        else:
                            raise Exception('Only "tandem" or "parallel" is valid for parameter arrangement_mode!')
                counter += 1
        f_cid2frag_sentence.close()
        if None in lines:
            for _, file_handle in file_handles.items():
                file_handle.close()
            break

    # count fragment frequency
    print('Start to count fragments...')
    if not os.path.exists(result_file_frag2num):
        frag2num_recount = count_fragment(result_file_cid2frag)
        frag2num_recount.to_csv(result_file_frag2num, index_label='frag_id')
    else:
        print('>>> frag2num_count file has existed, this step will omit...')
        print()
    # get fragment sentence
    print('Start to get fragment sentence separated by space...')
    if not os.path.exists(result_file_frag_smiles_sentence):
        result_file_cid2frag_id = os.path.join(result_dir, '{}_cid2frag_id_sentence.csv'.format(result_file_prefix))
        get_fragment_sentence(result_file_frag2num, result_file_cid2frag,
                              result_fp=result_file_cid2frag_id,
                              result_fp2=result_file_frag_smiles_sentence, replace_by_id=False)
    else:
        print('>>> frag_smiles_sentence file has existed, this step will omit...')
        print()


def get_tandem_by_parallel(cid2frag_smiles_file_path, result_dir):
    """
    generate tandem fragment sentence according to parallel result
    :param cid2frag_smiles_file_path: cid2frag_smiles file of parallel model
    :param result_dir:
    :return:
    """
    # frag_info_files = ['cid2mol_smiles.txt', 'cid2edges.txt', 'cid2frag_id2mol_inx.txt']
    result_file_prefix = 'tandem'
    result_file_cid2frag = os.path.join(result_dir, '{}_cid2smiles_sentence.csv'.format(result_file_prefix))
    # result_file_frag2num = os.path.join(result_dir, '{}_frag2num_count.csv'.format(result_file_prefix))
    # frag_id or frag_smiles
    result_file_frag_smiles_sentence = os.path.join(result_dir, '{}_frag_smiles_sentence.csv'.format(result_file_prefix))

    with open(cid2frag_smiles_file_path, 'r', encoding='utf-8') as parallel_f:
        cid2frag_smiles = {}
        # cids = []
        cid_counter = 0
        cid2sentence_f = open(result_file_cid2frag, 'a', encoding='utf-8')
        sentence_f = open(result_file_frag_smiles_sentence, 'a', encoding='utf-8')
        for line in tqdm(parallel_f):
            cid, frag_smiles = line.strip().split('\t')
            if cid not in cid2frag_smiles:
                for _cid, _frag_smiles_list in cid2frag_smiles.items():
                    # _cid is the previous CID instead of current "cid"
                    cid_counter += 1
                    mol_sentence = ','.join(_frag_smiles_list)
                    cid2sentence_f.write('\t'.join([_cid, mol_sentence]) + '\n')
                    _mol_sentence = mol_sentence.split(',')
                    sentence_f.write(' '.join(_mol_sentence) + '\n')
                # cids = []
                cid2frag_smiles = {cid: []}
                # cids.append(cid)
            cid2frag_smiles[cid].append(frag_smiles)
        print('>>> CID counter: {}'.format(cid_counter))
        print('>>> Output the last one...')
        for _cid, _frag_smiles_list in cid2frag_smiles.items():
            # _cid is the previous CID instead of current "cid"
            cid_counter += 1
            mol_sentence = ','.join(_frag_smiles_list)
            cid2sentence_f.write('\t'.join([_cid, mol_sentence]) + '\n')
            _mol_sentence = mol_sentence.split(',')
            sentence_f.write(' '.join(_mol_sentence) + '\n')
        cid2sentence_f.close()
        sentence_f.close()


if __name__ == '__main__':
    testLines = 10
    min_len = 5
    parser = argparse.ArgumentParser(
        description='molecule fragment by tree decomposition')
    parser.add_argument('input_fn',
                        help='file path of molecular fragments information from the first step, '
                             'which contains cid/smiles/frag_id2smiles/frag_id2neighbors/frag_id2mol_inx')
    parser.add_argument('result_dir',
                        help='result directory')
    parser.add_argument('--arrangement_mode',
                        help='how to arrange fragments in different molecular paths '
                             'of a single molecule, tandem/parallel (parallel is better)',
                        default='parallel')
    parser.add_argument('--refragment', action='store_true', default=False, help='whether to use raw fragments '
                                                                                 'or re-fragment')
    parser.add_argument('--frag2num_fp', help='file path of fragment2number from '
                                              'step1 result (fragment/count/frequency)',
                        default='no_input')
    parser.add_argument('--log_fn',
                        help='log file name')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Run the entire script on only the first {} lines and plot.'.format(testLines))
    # args = parser.parse_args()
    # test = args.test
    # arrangement_mode = args.arrangement_mode
    # refragment = args.refragment
    # f2n_fp = args.frag2num_fp
    # log_file = args.log_fn
    # input_file = args.input_fn
    # result_dir = args.result_dir

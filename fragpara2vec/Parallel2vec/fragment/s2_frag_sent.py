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
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
from .pub_func import write_list_by_json
from .helper_func import Mol2Network, get_fragment_sentence, count_fragment, Refragment


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


def call_frag2sent(input_file_dir, log_file, arrangement_mode='parallel',
                   refragment=False, frag2num_fp=None, result_dir=None, test=False):
    """
    TODO: remove refragment or improve refragment, cannot work now, 9.26, 2020
    get fragment sentence after fragmentation by tree decomposition
    :param input_file_dir: file dir of molecular fragments information from the first step,
        which contains four files:
         cid2mol_smiles.txt:  cid/smiles
         cid2frag_id2frag_smiles.txt: cid/frag_id2smiles
         cid2frag_id2neighbors: cid/frag_id2neighbors
         cid2frag_id2mol_inx.txt: cid/smiles/frag_id2mol_inx
    :param log_file: log file name
    :param arrangement_mode: how to arrange fragments in different molecular paths of a single molecule,
        tandem/parallel (parallel is better)
    :param refragment: whether to use raw fragments or re-fragment
    :param frag2num_fp: file path of fragment2number from the first step (fragment/count/frequency)
    :param result_dir: result directory
    :param test: Run the entire script on only the first 10 lines and plot
    :return:
    """
    frag_info_files = ['cid2mol_smiles.txt', 'cid2frag_id2frag_smiles.txt',
                       'cid2frag_id2neighbors.txt', 'cid2frag_id2mol_inx.txt']
    testLines = 10
    # min_len = 5
    result_file_prefix = 'tandem'
    frag2num = pd.DataFrame()
    if arrangement_mode == 'parallel':
        result_file_prefix = 'parallel'
    if refragment:
        # TODO: need to generate fragment2frequency before this step
        print('Refragment will be execute...')
        result_file_prefix += '_refrag'
        if frag2num_fp == 'no_input':
            raise Exception('The file path of fragment2frequency is needed, '
                            'please set the parameter of --frag2freq_fp')
        else:
            frag2num = pd.read_csv(frag2num_fp, index_col=0)
    print('Current arrangement mode is: {}'.format(arrangement_mode))
    result_file_cid2frag = os.path.join(result_dir, '{}_cid2smiles_sentence.csv'.format(result_file_prefix))
    result_file_frag2num = os.path.join(result_dir, '{}_frag2num_count.csv'.format(result_file_prefix))
    # frag_id or frag_smiles
    result_file_frag_smiles_sentence = os.path.join(result_dir, '{}_frag_smiles_sentence.csv'.format(result_file_prefix))

    result_file_id2frag_refrag = os.path.join(result_dir, 'id2frag_info_refrag.csv')
    test_fig_dir = ''
    if test:
        test_fig_dir = os.path.join(result_dir, 'test', 'figure')
        if not os.path.exists(test_fig_dir):
            os.makedirs(test_fig_dir)

    print('Start to generate fragment sentence by molecular tree...')
    if not os.path.exists(result_file_cid2frag):
        file_handles = {filename: open(os.path.join(input_file_dir, filename), 'r') for filename in frag_info_files}
        counter = 0
        while 1:
            # https://stackoverflow.com/a/46392753/2803344
            if test and counter > testLines:
                break
            current_row = []
            line = None
            for filename, file in file_handles.items():
                line = next(file, None)
                if line is not None:
                    line = line.strip().split('\t')
                    if line[0] == 'cid':
                        if filename == 'cid2mol_smiles.txt':
                            current_row += line
                        else:
                            current_row += line[1:]
                    else:
                        if filename == 'cid2mol_smiles.txt':
                            current_row += [json.loads(i) for i in line]
                        else:
                            current_row += [json.loads(i) for i in line[1:]]
                else:
                    file.close()
            if line is None:
                break
            counter += 1
            cid = current_row[0]
            if cid != 'cid':
                SMILES = current_row[1]
                id2smiles = {int(i): j for i, j in current_row[2].items()}  # frag_id2smiles
                n2n = {int(i): j for i, j in current_row[3].items()}  # frag_id2neighbors
                id2mol_inx = {int(i): j for i, j in current_row[4].items()}  # frag_id2mol_inx
                if counter % 100000 == 0 or test:
                    print('>>> current line: ', counter)
                    print('>>>CID: {}, SMILES: (line {})'.format(cid, counter), SMILES)
                network = Mol2Network(smiles=SMILES, n2n=n2n, id2smiles=id2smiles, id2mol_inx=id2mol_inx)
                if refragment and (not os.path.exists(result_file_id2frag_refrag)):
                    refragment_class = Refragment(network.g, f2f=frag2num, smiles=SMILES)
                    refragment_result = refragment_class.update()
                    frag2num = refragment_result['f2f']  # update f2n at each iterate
                    n2n = refragment_result['n2n']  # update n2n after refragment
                    id2smiles = refragment_result['id2smiles']
                    id2mol_inx = refragment_result['id2mol_inx']
                    network = Mol2Network(smiles=SMILES, n2n=n2n, id2smiles=id2smiles,
                                          id2mol_inx=id2mol_inx)
                    with open(result_file_id2frag_refrag, 'a') as result_f_refrag:
                        write_str = write_list_by_json([cid, SMILES, id2smiles, n2n, id2mol_inx])
                        result_f_refrag.write(write_str)
                if test:
                    print('    id2smiles: ', id2smiles)
                    print('    n2n: ', n2n)
                    draw_graph(network.g, file_dir=test_fig_dir,
                               file_name='mol_tree_cid:{}_line:{}'.format(cid, counter))
                    mol = Chem.MolFromSmiles(SMILES)
                    Draw.MolToFile(mol, os.path.join('big-data', 'figure',
                                                     'mol_structure_{}.png'.format(counter)))
                    mol_with_inx = mol_with_atom_index(mol)
                    Draw.MolToFile(mol_with_inx,
                                   os.path.join('big-data', 'figure',
                                                'mol_with_inx_{}.png'.format(counter)))
                mol_paths_smiles = []
                try:
                    mol_paths = network.get_mol_path()  # all mol paths, a list of lists
                    if counter % 100000 == 0:
                        print('>>> current line: ', counter)
                        print('    The longest mol path of this molecule: ', mol_paths[0])
                    for mol_path in mol_paths:  # replace frag_id by smiles
                        # remove duplicated items, Aug 7, 2020
                        mol_path_by_smiles = [id2smiles[frag_id] for frag_id in mol_path]
                        if mol_path_by_smiles not in mol_paths_smiles:
                            mol_paths_smiles.append(mol_path_by_smiles)
                except Exception as e:
                    mol_paths = []
                    with open(log_file, 'a') as log_f:
                        log_f.write('getting mol path from network has error, cid: {}'.format(cid) + '\n')
                if mol_paths_smiles:
                    mol_sentence = []
                    with open(result_file_cid2frag, 'a') as f:
                        # mol_path_str = ','.join([str(i) for i in mol_paths])
                        if arrangement_mode == 'tandem':
                            for mol_ps in mol_paths_smiles:
                                mol_sentence += mol_ps
                            frag_smiles = ','.join(mol_sentence)
                            f.write('\t'.join([cid, frag_smiles]) + '\n')
                        elif arrangement_mode == 'parallel':
                            for mol_ps in mol_paths_smiles:
                                # may have multiple sentences for each molecule
                                one_frag_smiles = '\t'.join([cid, ','.join(mol_ps)])
                                mol_sentence.append(one_frag_smiles)
                            f.write('\n'.join(mol_sentence) + '\n')
                            # for mol_s in mol_sentence:
                            #     frag_smiles = ','.join(mol_s)
                        else:
                            raise Exception('Only "tandem" or "parallel" is valid for parameter arrangement_mode!')
                        if counter % 100000 == 0 or test:
                            # print('>>> current line: ', counter)
                            print('    Mol sentence: ')
                            print(mol_sentence)
                            print('    mol paths: ', mol_paths)
                    # print(line)

        if refragment:
            frag2num.to_csv(os.path.join(result_dir, 'step2_frag2num_new.csv'))
    else:
        print('>>> cid2smiles_sentence file has existed, this step will omit...')
        print()

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

    # with open(input_file_dir, 'r') as input_f:
    #     counter = 0
    #     for row in input_f:
    #         if test and counter > testLines:
    #             break
    #         row = row.strip().split('\t')
    #         if row[0] == 'cid':
    #             current_row = row
    #         else:
    #             current_row = [json.loads(i) for i in row]
    #         try:
    #             cid = current_row[0]
    #         except KeyError:
    #             cid = ''
    #         if row[0] != 'cid':  # remove the first line
    #             SMILES = current_row[1]
    #             id2smiles = {int(i): j for i, j in current_row[2].items()}  # frag_id2smiles
    #             n2n = {int(i): j for i, j in current_row[3].items()}  # frag_id2neighbors
    #             id2mol_inx = {int(i): j for i, j in current_row[4].items()}  # frag_id2mol_inx
    #             if counter % 100000 == 0 or test:
    #                 print('>>> current line: ', counter)
    #                 print('>>>CID: {}, SMILES: (line {})'.format(cid, counter), SMILES)
    #             network = Mol2Network(smiles=SMILES, n2n=n2n, id2smiles=id2smiles, id2mol_inx=id2mol_inx)
    #             if refragment:
    #                 refragment_class = Refragment(network.g, f2f=frag2num, smiles=SMILES)
    #                 refragment_result = refragment_class.update()
    #                 frag2num = refragment_result['f2f']  # update f2n at each iterate
    #                 n2n = refragment_result['n2n']  # update n2n after refragment
    #                 id2smiles = refragment_result['id2smiles']
    #                 id2mol_inx = refragment_result['id2mol_inx']
    #                 network = Mol2Network(smiles=SMILES, n2n=n2n, id2smiles=id2smiles,
    #                                       id2mol_inx=id2mol_inx)
    #                 if output_id2frag_refrag:
    #                     with open(result_file_id2frag_refrag, 'a') as result_f_refrag:
    #                         write_str = write_list_by_json([cid, SMILES, id2smiles, n2n, id2mol_inx])
    #                         result_f_refrag.write(write_str)
    #
    #             if test:
    #                 print('    id2smiles: ', id2smiles)
    #                 print('    n2n: ', n2n)
    #                 draw_graph(network.g, file_dir=test_fig_dir,
    #                            file_name='mol_tree_cid:{}_line:{}'.format(cid, counter))
    #                 mol = Chem.MolFromSmiles(SMILES)
    #                 Draw.MolToFile(mol, os.path.join('big-data', 'figure',
    #                                                  'mol_structure_{}.png'.format(counter)))
    #                 mol_with_inx = mol_with_atom_index(mol)
    #                 Draw.MolToFile(mol_with_inx,
    #                                os.path.join('big-data', 'figure',
    #                                             'mol_with_inx_{}.png'.format(counter)))
    #
    #             mol_paths_smiles = []
    #             try:
    #                 mol_paths = network.get_mol_path()  # all mol paths, a list of lists
    #                 if counter % 100000 == 0:
    #                     print('>>> current line: ', counter)
    #                     print('    The longest mol path of this molecule: ', mol_paths[0])
    #                 for mol_path in mol_paths:  # replace frag_id by smiles
    #                     # remove duplicated items, Aug 7, 2020
    #                     mol_path_by_smiles = [id2smiles[frag_id] for frag_id in mol_path]
    #                     if mol_path_by_smiles not in mol_paths_smiles:
    #                         mol_paths_smiles.append(mol_path_by_smiles)
    #             except Exception as e:
    #                 with open(log_file, 'a') as log_f:
    #                     log_f.write('getting mol path from network has error, cid: {}'.format(cid) + '\n')
    #
    #             if mol_paths_smiles:
    #                 mol_sentence = []
    #
    #                 with open(result_file_cid2frag, 'a') as f:
    #                     # mol_path_str = ','.join([str(i) for i in mol_paths])
    #                     if arrangement_mode == 'tandem':
    #                         for mol_ps in mol_paths_smiles:
    #                             mol_sentence += mol_ps
    #                         frag_smiles = ','.join(mol_sentence)
    #                         f.write('\t'.join([cid, frag_smiles]) + '\n')
    #                     elif arrangement_mode == 'parallel':
    #                         for mol_ps in mol_paths_smiles:
    #                             # may have multiple sentences for each molecule
    #                             one_frag_smiles = '\t'.join([cid, ','.join(mol_ps)])
    #                             mol_sentence.append(one_frag_smiles)
    #                         f.write('\n'.join(mol_sentence) + '\n')
    #                         # for mol_s in mol_sentence:
    #                         #     frag_smiles = ','.join(mol_s)
    #                     else:
    #                         raise Exception('Only "tandem" or "parallel" is valid for parameter arrangement_mode!')
    #                     if counter % 100000 == 0 or test:
    #                         # print('>>> current line: ', counter)
    #                         print('    Mol sentence: ')
    #                         print(mol_sentence)
    #                         print('    mol paths: ', mol_paths)
    #         counter += 1
    # if refragment:
    #     frag2num.to_csv(os.path.join(result_dir, 'step2_frag2num_new.csv'))
    #
    # # count fragment frequency
    # print('>>> Start to count fragments...')
    # frag2num_recount = count_fragment(result_file_cid2frag)
    # frag2num_recount.to_csv(result_file_frag2num, index_label='frag_id')
    #
    # # get fragment sentence
    # print('>>> Start to get sentence separated by space...')
    # get_fragment_sentence(result_file_frag2num, result_file_cid2frag,
    #                       result_fp=result_file_cid2frag_id,
    #                       result_fp2=result_file_frag_smiles_sentence, replace_by_id=False)


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
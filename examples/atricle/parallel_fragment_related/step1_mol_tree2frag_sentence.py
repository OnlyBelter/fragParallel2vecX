import os
import pkg_resources
from fragpara2vec.Parallel2vec import call_mol_tree, call_frag2sent


def split_mol_tree(file_path, result_dir):
    """

    :param file_path:
    :return:
    """
    columns = ['mol_smiles', 'frag_id2frag_smiles', 'frag_id2neighbors', 'frag_id2mol_inx']
    file_names = ['cid2' + i + '.txt' for i in columns]
    # result = pd.DataFrame()
    for i in range(len(columns)):
        if not os.path.exists(os.path.join(result_dir, file_names[i])):
            with open(os.path.join(result_dir, file_names[i]), 'w') as f:
                f.write('\t'.join(['cid', columns[i]]) + '\n')
    with open(file_path) as f:
        counter = 0
        for line in f:
            counter += 1
            _line = line.strip().split('\t')
            cid = _line[0]
            if cid.lower() != 'cid':
                smiles = _line[1]
                mol_blocks = _line[2]
                node2neighbors = _line[3]
                id2mol_inx = _line[4]
                col2val = {columns[0]: smiles, columns[1]: mol_blocks,
                           columns[2]: node2neighbors, columns[3]: id2mol_inx}
                for j in range(len(columns)):
                    with open(os.path.join(result_dir, file_names[j]), 'a') as result_f:
                        write_str = '\t'.join([cid, col2val[columns[j]]]) + '\n'
                        result_f.write(write_str)
        print('>>> There are {} lines in total.'.format(counter))


if __name__ == '__main__':
    # root_dir = pkg_resources.resource_filename('fragpara2vec', 'demo_data')
    root_dir = '../../../big_data'
    sub_dir1 = '02_filtered_molecule'
    sub_dir2 = '03_fragment'
    sub_dir3 = 'cid2frag_info'
    raw_data_file_path = os.path.join(root_dir, sub_dir1, 'cid2SMILES_after_filtered.txt')
    result_dir = os.path.join(root_dir, sub_dir2)

    # tree decomposition
    print('Start to do tree decomposition...')
    tree_decompo_result_dir = os.path.join(result_dir, sub_dir3)
    if not os.path.exists(tree_decompo_result_dir):
        call_mol_tree(raw_data_file=raw_data_file_path, log_file='mol_tree.log', result_dir=tree_decompo_result_dir)
    else:
        print('>>> The result folder of tree decomposition has existed, previous results will be used.')
        print()
    # for mol_tree_file in ['mol_tree.txt', 'mol_tree_part2.txt']:
    #     split_mol_tree(os.path.join(result_dir, mol_tree_file), result_dir)

    # get fragment sentence from decomposition results
    for arrange_mode in ['parallel', 'tandem']:
        print('Start to generate fragment sentence by {} arrangement mode'.format(arrange_mode))
        call_frag2sent(input_file_dir=tree_decompo_result_dir, log_file='frag2sent_{}.log'.format(arrange_mode),
                       arrangement_mode=arrange_mode, result_dir=result_dir)

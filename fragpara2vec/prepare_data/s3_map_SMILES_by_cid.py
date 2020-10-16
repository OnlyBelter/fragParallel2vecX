import os


def query_cid2smiles_by_cid(filtered_mol_file_path, all_cid2smiles_file_path, output_file_path):
    """

    :param filtered_mol_file_path:
    :param all_cid2smiles_file_path:
    :param output_file_path:
    :return:
    """
    if not os.path.exists(output_file_path):
        with open(output_file_path, 'w') as file_h:
            file_h.write('\t'.join(['cid', 'smiles']) + '\n')
    cid2smiles = {}
    with open(all_cid2smiles_file_path, 'r') as file_handle1:
        for each_line in file_handle1:
            each_line = each_line.strip()
            cid, smiles = each_line.split('\t')
            cid2smiles[cid] = smiles
    with open(filtered_mol_file_path, 'r') as file_handle2:
        for each_line2 in file_handle2:
            each_line2 = each_line2.strip()
            cid, _, _, _ = each_line2.split(',')
            with open(output_file_path, 'a') as file_handle3:
                file_handle3.write('\t'.join([cid, cid2smiles[cid]]) + '\n')


if __name__ == '__main__':
    root_dir = '../../big_data'
    # filtered_mol_fn = 'cid2md_filtered_by_md.txt'
    # all_cid2smiles_fn = 'cid2SMILES.txt'
    # all_cid2smiles_fp = os.path.join(root_dir, '01_raw_data', all_cid2smiles_fn)
    # filtered_mol_fp = os.path.join(root_dir, '02_filtered_molecule', filtered_mol_fn)
    # output_fp = os.path.join(root_dir, '02_filtered_molecule', 'cid2SMILES_filtered.txt')


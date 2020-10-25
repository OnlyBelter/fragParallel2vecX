import os
from fragpara2vec.prepare_data import (call_download_cid2smiles, remove_duplicate_smiles,
                                       do_step2, filter_by_md, query_cid2smiles_by_cid)


if __name__ == '__main__':
    root_dir = r'F:\result_dir\test4'
    sub_dir = '01_raw_data'
    sub_dir2 = '02_filtered_molecule'
    result_dir_step1 = os.path.join(root_dir, sub_dir)
    if not os.path.exists(result_dir_step1):
        os.makedirs(result_dir_step1)
    cid_list = [28376780, 16352706, 8536751, 18386381, 9085153, 24320025, 8626124, 12012895, 3333305,
                14187785, 5287182, 20393803, 21401959, 15480910, 25007344, 18029384, 4384958, 6468823,
                24226358, 19375211]
    interval = 200
    cid2smiles_file_name = 'cid2SMILES.txt'
    cid2smiles_file_path = os.path.join(result_dir_step1, cid2smiles_file_name)
    removed_duplicate_smiles_file_path = os.path.join(result_dir_step1, 'cid2SMILES_removed_duplicates.txt')
    if not os.path.exists(cid2smiles_file_path):
        call_download_cid2smiles(interval=interval, cid_list=cid_list,
                                 result_file_dir=result_dir_step1, file_name=cid2smiles_file_name)
        remove_duplicate_smiles(cid2smiles_file_path=cid2smiles_file_path,
                                output_file_path=removed_duplicate_smiles_file_path)

    # filter downloaded result by MD
    result_dir_step2 = os.path.join(root_dir, sub_dir2)
    if not os.path.exists(result_dir_step2):
        os.makedirs(result_dir_step2)

    output_fp = os.path.join(result_dir_step2, 'cid2md_filtered_by_ele_and_charge.txt')
    output_fp_filtered_by_md = os.path.join(result_dir_step2, 'cid2md_filtered_by_md.txt')
    do_step2(input_fp=removed_duplicate_smiles_file_path, output_fp=output_fp)
    filter_by_md(input_fp=output_fp, output_fp=output_fp_filtered_by_md)

    # query cid2smiles
    query_cid2smiles_by_cid(filtered_mol_file_path=output_fp_filtered_by_md,
                            all_cid2smiles_file_path=removed_duplicate_smiles_file_path,
                            output_file_path=os.path.join(result_dir_step2, 'cid2SMILES_filtered.txt'))

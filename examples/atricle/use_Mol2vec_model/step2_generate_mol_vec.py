import os
from fragpara2vec.utility import get_mol_vec
from fragpara2vec.Mol2vec import generate_corpus_from_smiles


if __name__ == '__main__':
    model_type = 'regression'  # or classification, regression is better
    root_dir = r'F:\github\fragParallel2vecX\big_data'
    # subdir_fragment = r'03_fragment/cid2frag_info'
    subdir1 = '07_sampling'
    subdir2 = '08_train_mol_vec_by_mlp'
    subdir_tandem = '05_model_Tandem2vec'
    subdir_parallel = '06_model_Parallel2vec'
    subdir_Mol2vec = '04_model_Mol2vec'
    cid2smiles_file = 'cid2mol_smiles.txt'
    result_dir = os.path.join(root_dir, subdir2)
    down_sampled_mol_file = 'selected_cid2md_class.csv'
    selected_cid2smiles_file_path = os.path.join(result_dir, 'selected_cid2smiles.csv')

    # fragmentation by fingerprinting
    cid2frag_fingerprint_dir = os.path.join(result_dir, 'fingerprinting')
    if not os.path.exists(cid2frag_fingerprint_dir):
        os.makedirs(cid2frag_fingerprint_dir)
    cid2frag_fingerprint_file_path = os.path.join(cid2frag_fingerprint_dir, 'cid2frag_id.txt')
    if not os.path.exists(cid2frag_fingerprint_file_path):
        print('Start to generate fragments by fingerprinting...')
        generate_corpus_from_smiles(in_file=selected_cid2smiles_file_path,
                                    out_file=cid2frag_fingerprint_file_path,
                                    r=1, n_jobs=6, keep_cid=True)
    mol_vector_file_path2 = os.path.join(cid2frag_fingerprint_dir, 'mol_vec_Mol2vec_frag.csv')
    if not os.path.exists(mol_vector_file_path2):
        print('>>> generate mol2vec by Mol2vc model...')
        frag_vec_Mol2vec_file_path = os.path.join(root_dir, subdir_Mol2vec, 'Mol2vec_frag_id2vec.csv')
        get_mol_vec(frag2vec_file_path=frag_vec_Mol2vec_file_path,
                    data_set=cid2frag_fingerprint_file_path,
                    result_path=mol_vector_file_path2)

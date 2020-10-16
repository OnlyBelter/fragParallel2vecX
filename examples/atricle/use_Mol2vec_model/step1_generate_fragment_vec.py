import os
from fragpara2vec.utility import insert_unk, count_frag_in_mol_sentence
from fragpara2vec.Mol2vec import (generate_corpus_from_smiles, train_word2vec_model,
                                  get_frag_vector_word2vec)


if __name__ == '__main__':
    # use all SMILES in MOSES data-set to train mol2vec model
    # file_path = r'F:\github\fragTandem2vecX\big_data\02_filtered_molecule\cid2SMILES_filtered.txt'
    root_dir = '../../../big_data'
    # cid_list = '../big-data/all_cid2smiles/step5_x_training_set.csv'
    sub_dir1 = '04_model_Mol2vec'
    # result_file_path1 = os.path.join('../big-data/moses_dataset/nn/parallel/cid2smiles_all_in_train_test.csv')
    sub_dir2 = '02_filtered_molecule'
    cid2smiles_fp = os.path.join(root_dir, sub_dir2, 'cid2SMILES_filtered.txt')
    result_file_path2 = os.path.join(root_dir, sub_dir1, 'cid2smiles_training_set_coupus.tmp')
    result_file_path3 = os.path.join(root_dir, sub_dir1, 'cid2smiles_training_set_coupus.txt')
    result_file_frag2num = os.path.join(root_dir, sub_dir1, 'Mol2vec_frag_id2num.csv')
    model_fp = os.path.join(root_dir, sub_dir1, 'Mol2vec_model.pkl')

    # step1 generate corpus (sentence)
    print('>>> Generate corpus...')
    if not os.path.exists(result_file_path2):
        generate_corpus_from_smiles(in_file=cid2smiles_fp, out_file=result_file_path2, r=1, n_jobs=6)

    # step2 Handling of uncommon "words"
    print('>>> Handle uncommon fragments...')
    if not os.path.exists(result_file_path3):
        insert_unk(corpus=result_file_path2, out_corpus=result_file_path3, threshold=3)

    # step3 count fragment in corpus (fragment sentence)
    print('>>> Count fragment in each fragment sentence...')
    if not os.path.exists(result_file_frag2num):
        frag2num = count_frag_in_mol_sentence(result_file_path3)
        frag2num.to_csv(result_file_frag2num)

    # step3 train molecule vector
    print('>>> Train fragment vector by Mol2vec model...')
    if not os.path.exists(model_fp):
        train_word2vec_model(infile_name=result_file_path3, outfile_name=model_fp,
                             vector_size=100, window=10, min_count=3, n_jobs=6, method='cbow')

    # step4 get fragment vector from pre-trained model
    print('>>> Get fragment vector from pre-trained Mol2vec model...')
    frag2vec_fp = os.path.join(root_dir, sub_dir1, 'Mol2vec_frag_id2vec.csv')
    if not os.path.exists(frag2vec_fp):
        get_frag_vector_word2vec(model_fp=model_fp, frag_id2vec_fp=frag2vec_fp)

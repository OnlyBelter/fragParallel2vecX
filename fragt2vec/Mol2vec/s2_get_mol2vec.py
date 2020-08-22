# down-sampling for what???
import os
import pandas as pd
from gensim.models import word2vec
from .helper_func import sentences2vec, DfVec, MolSentence


def get_cid2smiles(cid2smiles, cid_list, result_file):
    """
    get cid2smiles.txt in training set
    :param cid2smiles: file path, cid, smiles
    :param cid_list： file path of cid list in train_set
    :param result_file: file path of result
    :return:
    """
    cid2smiles_dict = {}
    with open(cid2smiles, 'r') as f_handle:
        for i in f_handle:
            i = i.strip()
            cid, smiles = i.split('\t')
            cid2smiles_dict[cid] = smiles
    with open(cid_list, 'r') as f_handle2:
        for i in f_handle2:
            i = i.strip()
            cid = i.split('\t')[0]
            if cid in cid2smiles_dict:
                with open(result_file, 'a') as r_handle:
                    r_handle.write(cid + '\t' + cid2smiles_dict[cid] + '\n')
            else:
                print('>>> this compound {} does not exist in our cid2smiles.txt list...'.format(cid))


def load_trained_model(model_fp):
    """
    load well-trained model by following function
    train_word2vec_model('./mols_demo_corpus.txt', outfile_name='mols_demo_model.pkl',
                          vector_size=150, window=10, min_count=3, n_jobs=2, method='skip-gram')
    :param model_fp:
    :return:
    """
    model = word2vec.Word2Vec.load(model_fp)
    return model


if __name__ == '__main__':
    # downsampled couspus, as order as file in result_file_path1
    dowmsampled_coupus_fp = os.path.join(root_dir, 'downsampled', 'cid2smiles_training_set_coupus.txt')
    mol2vec_fp = os.path.join(root_dir, 'model_mol2vec_mol2vec_trained_by_all_MOSES.csv')
    cid2smiles_test = '../big-data/cid2smiles_test.txt'
    result_file_path4 = '../big-data/vectors/mol2vec_model_mol2vec.csv'
    get_cid2smiles(cid2smiles.txt, cid_list, result_file=reuslt_file_path1)

    # get vector of each molecule by mol2vec model
    # mol with fragment id sentence
    print('Start to read downsampled mol sentences and load model...')
    mol_info = pd.read_csv(dowmsampled_coupus_fp, header=None)

    # model_fp = os.path.join(include_small_dataset_dir, 'mol2vec_related', 'mol2vec_model.pkl')
    model = load_trained_model(model_fp)
    # print(mol_info.loc[4568802, '0'])
    mol_info['sentence'] = mol_info.apply(lambda x: MolSentence([str(i) for i in x[0].split(' ')]), axis=1)
    # print(mol_info)
    mol_info['mol2vec_related'] = [DfVec(x) for x in sentences2vec(mol_info['sentence'], model)]
    cid2vec = {}
    cid2smiles = pd.read_csv(result_file_path1)
    inx2cid = cid2smiles['0'].to_dict()
    for inx in mol_info.index.to_list():
        cid = inx2cid[inx]
        cid2vec[cid] = list(mol_info.loc[inx, 'mol2vec_related'].vec)
    cid2vec_df = pd.DataFrame.from_dict(cid2vec, orient='index')
    print(cid2vec_df.shape)
    # result_file2 = os.path.join(result_dir, 'step4_selected_mol2vec_model_mol2vec.csv')
    cid2vec_df.to_csv(mol2vec_fp, header=False, float_format='%.3f')

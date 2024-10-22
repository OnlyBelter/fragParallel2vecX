"""
train molecular vectors by Mol2vec model
"""
import os
import pandas as pd
import json
from tqdm import tqdm
from rdkit import Chem
from ..utility import insert_unk
from gensim.models import word2vec
from joblib import Parallel, delayed
from .helper_func import train_word2vec_model, mol2alt_sentence


def generate_corpus_from_smiles(in_file, out_file, r=1, sentence_type='alt', n_jobs=1, keep_cid=False):
    """
    modified from generate_corpus
    https://mol2vec.readthedocs.io/en/latest/#mol2vec.features.generate_corpus
    :param in_file: cid, smiles
    :param out_file:
    :param r: int, Radius of morgan fingerprint
    :param sentence_type:
    :param n_jobs:
    :param keep_cid: whether keep cid and output it to result file
    :return:
    """
    all_cid = []
    cid2smiles = {}
    with open(in_file, 'r') as f_handle:
        for each_line in tqdm(f_handle):
            if ',' in each_line:
                cid, smiles = each_line.strip().split(',')
            else:
                cid, smiles = each_line.strip().split('\t')
            if cid != 'cid':
                if '"' in smiles:
                    cid2smiles[cid] = json.loads(smiles)
                else:
                    cid2smiles[cid] = smiles
                all_cid.append(cid)
    print('>>> the number of unique cid is {}'.format(len(cid2smiles)))
    print('>>> the number of all cid is {}'.format(len(all_cid)))
    # assert len(smiles2cid) == len(all_smiles)

    if sentence_type == 'alt':  # This can run parallelized
        if keep_cid:
            result = Parallel(n_jobs=n_jobs, verbose=1)(delayed(_parallel_job_new)(cid, cid2smiles[cid], r)
                                                        for cid in all_cid)
        else:
            result = Parallel(n_jobs=n_jobs, verbose=1)(delayed(_parallel_job)(cid2smiles[cid], r) for cid in all_cid)
        for i, line in enumerate(result):
            with open(out_file, 'a') as f_handle:
                f_handle.write(str(line) + '\n')

        print('% molecules successfully processed.')


def _parallel_job_new(cid, smiles, r):
    """
    return result with cid
    :param cid:
    :param smiles:
    :param r:
    :return:
    """
    if smiles is not None:
        # smiles = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smiles)
        sentence = mol2alt_sentence(mol, r)
        sentence_str = ','.join(sentence)
        return "\t".join([cid, sentence_str])


def _parallel_job(smiles, r):
    """Helper function for joblib jobs
    """
    if smiles is not None:
        # smiles = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smiles)
        sentence = mol2alt_sentence(mol, r)
        return " ".join(sentence)


# def _read_corpus(file_name):
#     while True:
#         line = file_name.readline()
#         if not line:
#             break
#         yield line.split()


# def insert_unk(corpus, out_corpus, threshold=3, uncommon='UNK'):
#     """
#     Handling of uncommon "words" (i.e. identifiers).
#     It finds all least common identifiers (defined by threshold) and replaces them by 'uncommon' string.
#
#     Parameters
#     ----------
#     corpus : str
#         Input corpus file
#     out_corpus : str
#         Outfile corpus file
#     threshold : int
#         Number of identifier occurrences to consider it uncommon, <= this threshold
#     uncommon : str
#         String to use to replace uncommon words/identifiers
#
#     Returns
#     -------
#     """
#     # Find least common identifiers in corpus
#     f = open(corpus)
#     unique = {}
#     for i, x in tqdm(enumerate(_read_corpus(f)), desc='Counting identifiers in corpus'):
#         for identifier in x:
#             if identifier not in unique:
#                 unique[identifier] = 1
#             else:
#                 unique[identifier] += 1
#     n_lines = i + 1
#     least_common = set([x for x in unique if unique[x] <= threshold])
#     f.close()
#
#     f = open(corpus)
#     fw = open(out_corpus, mode='w')
#     for line in tqdm(_read_corpus(f), total=n_lines, desc='Inserting %s' % uncommon):
#         intersection = set(line) & least_common
#         if len(intersection) > 0:
#             new_line = []
#             for item in line:
#                 if item in least_common:
#                     new_line.append(uncommon)
#                 else:
#                     new_line.append(item)
#             fw.write(" ".join(new_line) + '\n')
#         else:
#             fw.write(" ".join(line) + '\n')
#     f.close()
#     fw.close()


def get_frag_vector_word2vec(model_fp, frag_id2vec_fp, frag_smiles=('all',)):
    """
    get fragment vector from pre-trained model by Word2vec
    https://radimrehurek.com/gensim/models/word2vec.html
    :param model_fp: file path of pre-trained model
    :param frag_id2vec_fp: file path of frag_id2vec
    :param frag_smiles: list
           get fragment vector of the fragments in this list, get all fragment vectors in model if ('all',)
    :return:
    """
    model = word2vec.Word2Vec.load(model_fp)
    words = model.wv.vocab
    # frag2vec = {}
    # for f in words:
    #     print(type(model.wv))
    #     # model.wv.vectors[1]
    #     frag2vec[f] = model.wv(f)
    frag2vec_df = pd.DataFrame(model.wv.vectors, index=model.wv.index2word)
    if len(frag_smiles) == 1 and frag_smiles[0] == 'all':
        pass
    elif len(frag_smiles) >= 1:
        frag2id = {}
        for smiles in frag_smiles:
            if smiles in words:
                frag2id[smiles] = words[smiles].index
            else:
                print('>>> Fragment SMILES {} does not exist in the model.'.format(smiles))
        frag2vec_df = frag2vec_df.loc[list(frag2id.keys()), :].copy()
    if frag_id2vec_fp:
        print('>>> There are {} fragments were returned.'.format(frag2vec_df.shape[0]))
        frag2vec_df.to_csv(frag_id2vec_fp, index_label='fragment')
    else:
        return frag2vec_df


if __name__ == '__main__':
    # use all SMILES in MOSES data-set to train mol2vec model
    # file_path = r'F:\github\fragTandem2vecX\big_data\02_filtered_molecule\cid2SMILES_filtered.txt'
    cid2smiles_fp = '../big_data/02_filtered_molecule/cid2SMILES_filtered.txt'
    # cid_list = '../big-data/all_cid2smiles/step5_x_training_set.csv'
    root_dir = '../../big_data/04_model_Mol2vec/'
    # result_file_path1 = os.path.join('../big-data/moses_dataset/nn/parallel/cid2smiles_all_in_train_test.csv')
    result_file_path2 = os.path.join(root_dir, 'cid2smiles_training_set_coupus.tmp')
    result_file_path3 = os.path.join(root_dir, 'cid2smiles_training_set_coupus.txt')
    model_fp = os.path.join(root_dir, 'mol2vec_model.pkl')

    # step1 generate corpus (sentence)
    generate_corpus_from_smiles(in_file=cid2smiles_fp, out_file=result_file_path2, r=1, n_jobs=6)

    # step2 Handling of uncommon "words"
    insert_unk(corpus=result_file_path2, out_corpus=result_file_path3)

    # step3 train molecule vector
    train_word2vec_model(infile_name=result_file_path3, outfile_name=model_fp,
                         vector_size=100, window=10, min_count=3, n_jobs=6, method='cbow')

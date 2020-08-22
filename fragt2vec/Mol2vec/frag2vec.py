"""
train molecular vectors by Mol2vec model
"""
import os
# from tqdm import tqdm
from rdkit import Chem
from joblib import Parallel, delayed
from .helper_func import train_word2vec_model, mol2alt_sentence
from ..utility import insert_unk


def generate_corpus_from_smiles(in_file, out_file, r, sentence_type='alt', n_jobs=1):
    """
    modified from generate_corpus
    https://mol2vec.readthedocs.io/en/latest/#mol2vec.features.generate_corpus
    :param in_file: cid, smiles
    :param out_file:
    :param r: int, Radius of morgan fingerprint
    :param sentence_type:
    :param n_jobs:
    :return:
    """
    all_smiles = []
    with open(in_file, 'r') as f_handle:
        for each_line in f_handle:
            if ',' in each_line:
                cid, smiles = each_line.strip().split(',')
            else:
                cid, smiles = each_line.strip().split('\t')
            if smiles != 'smiles':
                all_smiles.append(smiles)

    if sentence_type == 'alt':  # This can run parallelized
        result = Parallel(n_jobs=n_jobs, verbose=1)(delayed(_parallel_job)(smiles, r) for smiles in all_smiles)
        for i, line in enumerate(result):
            with open(out_file, 'a') as f_handle:
                f_handle.write(str(line) + '\n')
        print('% molecules successfully processed.')


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

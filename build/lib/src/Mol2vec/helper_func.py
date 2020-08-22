from tqdm import tqdm
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from gensim.models import word2vec
import timeit
from joblib import Parallel, delayed
import datetime


def mol2alt_sentence(mol, radius):
    """Same as mol2sentence() expect it only returns the alternating sentence
    Calculates ECFP (Morgan fingerprint) and returns identifiers of substructures as 'sentence' (string).
    Returns a tuple with 1) a list with sentence for each radius and 2) a sentence with identifiers from all radii
    combined.
    NOTE: Words are ALWAYS reordered according to atom order in the input mol object.
    NOTE: Due to the way how Morgan FPs are generated, number of identifiers at each radius is smaller

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    radius : float
        Fingerprint radius

    Returns
    -------
    list
        alternating sentence
    combined
    """
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)  # info: dictionary identifier, atom_idx, radius

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element  # {atom number: {fp radius: identifier}}

    # merge identifiers alternating radius to sentence: atom 0 radius0, atom 0 radius 1, etc.
    identifiers_alt = []
    for atom in dict_atoms:  # iterate over atoms
        for r in radii:  # iterate over radii
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(alternating_sentence)


def train_word2vec_model(infile_name, outfile_name=None, vector_size=100, window=10, min_count=3, n_jobs=1,
                         method='skip-gram', **kwargs):
    """Trains word2vec (Mol2vec, ProtVec) model on corpus file extracted from molecule/protein sequences.
    The corpus file is treated as LineSentence corpus (one sentence = one line, words separated by whitespaces)

    Parameters
    ----------
    infile_name : str
        Corpus file, e.g. proteins split in n-grams or compound identifier
    outfile_name : str
        Name of output file where word2vec model should be saved
    vector_size : int
        Number of dimensions of vector
    window : int
        Number of words considered as context
    min_count : int
        Number of occurrences a word should have to be considered in training
    n_jobs : int
        Number of cpu cores used for calculation
    method : str
        Method to use in model training. Options cbow and skip-gram, default: skip-gram)

    Returns
    -------
    word2vec.Word2Vec
    """
    if method.lower() == 'skip-gram':
        sg = 1
    elif method.lower() == 'cbow':
        sg = 0
    else:
        raise ValueError('skip-gram or cbow are only valid options')

    start = timeit.default_timer()
    corpus = word2vec.LineSentence(infile_name)
    model = word2vec.Word2Vec(corpus, size=vector_size, window=window, min_count=min_count, workers=n_jobs, sg=sg,
                              **kwargs)
    if outfile_name:
        model.save(outfile_name)

    stop = timeit.default_timer()
    print('Runtime: ', round((stop - start) / 60, 2), ' minutes')
    return model


def sentences2vec(sentences, model, unseen=None):
    """Generate vectors for each sentence (list) in a list of sentences. Vector is simply a
    sum of vectors for individual words.

    Parameters
    ----------
    sentences : list, array
        List with sentences
    model : word2vec.Word2Vec
        Gensim word2vec model
    unseen : None, str
        Keyword for unseen words. If None, those words are skipped.
        https://stats.stackexchange.com/questions/163005/how-to-set-the-dictionary-for-text-analysis-using-neural-networks/163032#163032

    Returns
    -------
    np.array
    """
    keys = set(model.wv.vocab.keys())
    vec = []
    if unseen:
        unseen_vec = model.wv.word_vec(unseen)

    for sentence in sentences:
        if unseen:
            vec.append(sum([model.wv.word_vec(y) if y in set(sentence) & keys
                            else unseen_vec for y in sentence]))
        else:
            vec.append(sum([model.wv.word_vec(y) for y in sentence
                            if y in set(sentence) & keys]))
    return np.array(vec)


class DfVec(object):
    """
    Helper class to store vectors in a pandas DataFrame

    Parameters
    ----------
    vec: np.array
    """

    def __init__(self, vec):
        self.vec = vec
        if type(self.vec) != np.ndarray:
            raise TypeError('numpy.ndarray expected, got %s' % type(self.vec))

    def __str__(self):
        return "%s dimensional vector" % str(self.vec.shape)

    __repr__ = __str__

    def __len__(self):
        return len(self.vec)

    _repr_html_ = __str__


class MolSentence:
    """Class for storing mol sentences in pandas DataFrame
    """
    def __init__(self, sentence):
        self.sentence = sentence
        if type(self.sentence[0]) != str:
            raise TypeError('List with strings expected')

    def __len__(self):
        return len(self.sentence)

    def __str__(self):  # String representation
        return 'MolSentence with %i words' % len(self.sentence)

    __repr__ = __str__  # Default representation

    def contains(self, word):
        """Contains (and __contains__) method enables usage of "'Word' in MolSentence"""
        if word in self.sentence:
            return True
        else:
            return False

    __contains__ = contains  # MolSentence.contains('word')

    def __iter__(self):  # Iterate over words (for word in MolSentence:...)
        for x in self.sentence:
            yield x

    _repr_html_ = __str__


def count_fragment(cid2frag_fp):
    """
    count fragment in all training set
    :param cid2frag_fp: cid2fragment file path, i.e. step2_result file
    :return:
    """
    frag2num = {}
    with open(cid2frag_fp, 'r') as f_handle:
        counter = 0
        for i in f_handle:
            if counter % 500000 == 0:
                t = datetime.datetime.now()
                print('>>> Current line: {}'.format(counter), t.strftime("%c"))
            cid, sentence = i.strip().split('\t')
            frags = sentence.split(',')
            for frag in frags:
                if frag not in frag2num:
                    frag2num[frag] = 0
                frag2num[frag] += 1
            counter += 1
    frag2num_df = pd.DataFrame.from_dict(frag2num, orient='index')
    frag2num_df.sort_values(by=0, inplace=True, ascending=False)
    frag2num_df.reset_index(inplace=True)
    frag2num_df.rename(columns={0: 'count', 'index': 'fragment'}, inplace=True)
    return frag2num_df

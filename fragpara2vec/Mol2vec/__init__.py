"""
model Mol2vec
https://github.com/samoturk/mol2vec

"""
from .helper_func import train_word2vec_model
from .frag2vec import generate_corpus_from_smiles
from .frag2vec import get_frag_vector_word2vec

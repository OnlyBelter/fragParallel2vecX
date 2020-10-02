"""
model Parallel2vec

"""
from .fragment.s1_mol_tree import call_mol_tree
from .fragment.s2_frag_sent import call_frag2sent
from .fragment.pub_func import write_list_by_json
from .parallel2vec import get_frag_vector_fasttext

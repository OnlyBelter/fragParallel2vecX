from .pub_func import get_mol
from .pub_func import get_mol_vec
from .pub_func import cal_md_by_smiles
from .pub_func import get_format_time
from .pub_func import count_frag_in_mol_sentence
from .pub_func import insert_unk
from .pub_func import reduce_by_tsne
from .pub_func import print_df
from .draw_mol import draw_multiple_mol, draw_mol_by_smiles, show_each_md
from .query_nearest_neighbors import cosine_dis, find_nearest_neighbor
from .compare_fragment import find_bond_pair
from .compare_fragment import find_aromatic_non_aroma_ring_pair
from .pub_func import SELECTED_MD
from .pub_func import MD_IMPORTANCE
from .pub_func import get_ordered_md
from .pub_func import grouper
from .pub_func import PUBCHEM_BASE_URL
from .pub_func import query_smiles_by_cids


MAIN_ELEMENT = ['C', 'O', 'N', 'H', 'P', 'S', 'Cl', 'F', 'Br', 'I']
# PUBCHEM_BASE_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/{}/CSV'
# ELEMENTS = ['S', 'Br', 'O', 'C', 'F', 'P', 'N', 'I', 'Cl', 'H']
BONDS = ['DOUBLE', 'SINGLE', 'TRIPLE']
PRMIER_NUM = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


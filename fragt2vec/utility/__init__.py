from .pub_func import get_mol
from .pub_func import cal_md_by_smiles
from .pub_func import get_format_time
from .pub_func import count_frag_in_mol_sentence
from .pub_func import insert_unk
from .draw_mol import draw_multiple_mol, draw_mol_by_smiles
from .query_nearest_neighbors import cosine_dis, find_nearest_neighbor

MAIN_ELEMENT = ['C', 'O', 'N', 'H', 'P', 'S', 'Cl', 'F', 'Br', 'I']
PUBCHEM_BASE_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/{}/CSV'
# ELEMENTS = ['S', 'Br', 'O', 'C', 'F', 'P', 'N', 'I', 'Cl', 'H']
BONDS = ['DOUBLE', 'SINGLE', 'TRIPLE']
SELECTED_MD = ['nN', 'nS', 'nO', 'nX', 'nBondsD', 'nBondsT', 'naRing', 'nARing']
PRMIER_NUM = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

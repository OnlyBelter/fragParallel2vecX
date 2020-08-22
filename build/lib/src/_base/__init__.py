from .pub_func import get_mol
from .pub_func import cal_md_by_smiles
from .pub_func import get_format_time

MAIN_ELEMENT = ['C', 'O', 'N', 'H', 'P', 'S', 'Cl', 'F', 'Br', 'I']
PUBCHEM_BASE_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/{}/CSV'
# ELEMENTS = ['S', 'Br', 'O', 'C', 'F', 'P', 'N', 'I', 'Cl', 'H']
BONDS = ['DOUBLE', 'SINGLE', 'TRIPLE']
SELECTED_MD = ['nN', 'nS', 'nO', 'nX', 'nBondsD', 'nBondsT', 'naRing', 'nARing']
PRMIER_NUM = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

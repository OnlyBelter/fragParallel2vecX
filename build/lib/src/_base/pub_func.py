import datetime
import pandas as pd
import rdkit.Chem as Chem
from mordred import Calculator, descriptors


def cal_md_by_smiles(smiles_list, md_list=None, print_info=False,
                     desc=(descriptors.AtomCount, descriptors.BondCount, descriptors.RingCount)):
    """
    calculate molecular descriptors by Mordred, https://github.com/mordred-descriptor/mordred
    :param smiles_list: a list of smiles
    :param md_list: a list of MD that need to output
    :param print_info:
    :param desc:
    :return:
    """
    SELECTED_MD = ['nN', 'nS', 'nO', 'nX', 'nBondsD', 'nBondsT', 'naRing', 'nARing']
    if print_info:
        print('  >There are {} SMILES in this list'.format(len(smiles_list)))
    calc = Calculator(desc, ignore_3D=True)
    mols = []
    for smiles in smiles_list:
        mols.append(Chem.MolFromSmiles(smiles))
    md_df = calc.pandas(mols)
    if not md_list:
        # naRing means aromatic ring count, nARing means aliphatic ring count
        md_list = SELECTED_MD
    if print_info:
        print(md_df)
        print(smiles_list)
    md_df['smiles'] = smiles_list
    md_df = md_df.loc[:, ['smiles'] + md_list].copy()
    if print_info:
        print('  >The shape of smiles_info is: {}'.format(md_df.shape))
    md_df.rename(columns={'smiles': 'fragment'}, inplace=True)
    md_df.set_index('fragment', inplace=True)
    return md_df


def get_mol(smiles):
    """
    SMILES -> mol obj
    :param smiles:
    :return:
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.Kekulize(mol)
    except ValueError:
        print('>>> Sanitization error: Can\'t kekulize mol: ', smiles)
        return None
    return mol


def print_df(df):
    assert type(df) == pd.core.frame.DataFrame
    print(df.shape)
    print(df.head(2))


def get_format_time():
    t = datetime.datetime.now()
    return t.strftime("%c")

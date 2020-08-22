import rdkit.Chem as Chem
from rdkit.Chem import Draw


def draw_mol_by_smiles(smiles, file_path=None):
    mol = Chem.MolFromSmiles(smiles)
    size = (200, 200)
    img = Draw.MolToImage(mol, size=size)
    if file_path:
        img.save(file_path)
    return img


def draw_multiple_mol(smiles_list, mols_per_row=4, file_path=None, legends=None):
    """
    draw molecular 2D structures
    :param smiles_list: a list of SMILES
    :param mols_per_row: int
    :param file_path: str, output file path
    :param legends: notes below each 2D structure
    :return:

    example:
    draw_multiple_mol(smiles_list=nn4['smiles'], legends=[str('{:.3f}'.format(i)) for i in nn4['dis']],
                      file_path='fig.svg')
    """
    mols = []
    for i in smiles_list:
        mols.append(Chem.MolFromSmiles(i))
    mols_per_row = min(len(smiles_list), mols_per_row)
    if legends is None:
        img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(220, 120), useSVG=True)
    else:
        img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(220, 120), useSVG=True, legends=legends)
    if file_path:
        with open(file_path, 'w') as f_handle:
            f_handle.write(img)
    return img

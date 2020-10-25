import rdkit.Chem as Chem
from rdkit.Chem import Draw
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns


def draw_mol_by_smiles(smiles, file_path=None):
    mol = Chem.MolFromSmiles(smiles)
    size = (200, 200)
    img = Draw.MolToImage(mol, size=size)
    if file_path:
        img.save(file_path)
    return img


def sys_print(output_str):
    # https://stackoverflow.com/a/12658698/2803344
    sys.stdout.write(output_str)  # same as print
    sys.stdout.flush()


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
    # print('>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<')
    for i in smiles_list:
        mols.append(Chem.MolFromSmiles(i))
    mols_per_row = min(len(smiles_list), mols_per_row)
    if legends is None:
        img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(220, 120), useSVG=True)
    else:
        img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(220, 120), useSVG=True, legends=legends)
    if file_path:
        pass
        # print(type(img))
        # with open(file_path, 'w') as f_handle:
        #     try:
        #         sys_print(type(img))
        #         f_handle.write(img)
        #     except TypeError:
        #         f_handle.write(img.data)
    return img


def show_each_md(x_reduced, frag_info, trim=False,
                 md_list=('nARing', 'naRing', 'nBondsT', 'nBondsD')):
    """
    :param x_reduced: 2 dimensions x with fragment as index, a dataframe
    :param frag_info: the number of each MD with fragment as index, a dataframe
    :param md_list: only 4 MD supported
    :param trim: only top 5 MD number will be showed
    """
    # model = model_name
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.flatten()
    # print(x_reduced.head(2))
    # print(frag_info.head(2))
    intersect_index = set(x_reduced.index.to_list()) & set(frag_info.index.to_list())
    x_reduced = x_reduced.loc[intersect_index, :].copy()  # alignment
    frag_info = frag_info.loc[intersect_index, md_list].copy()
    # reduced_x = reduced_x.loc[frag_info.index, :].copy()
    # parallel_frag_info = parallel_frag_info.loc[:, selected_md].copy()
    for i, md in enumerate(frag_info.columns.to_list()):
        # current_labels = parallel_frag_info.iloc[:, i]
        current_labels = frag_info.iloc[:, i]
        unique_labels = list(sorted(current_labels.unique()))
        if trim:
            if len(unique_labels) > 5:
                unique_labels = unique_labels[:5]
        n_labels = len(unique_labels)
        # print(n_labels)
        cc = sns.color_palette('Blues', n_labels)
        for j, label in enumerate(unique_labels):
            current_nodes = (current_labels == label)
            ax[i].scatter(x_reduced.loc[current_nodes, 0], x_reduced.loc[current_nodes, 1],
                          c=colors.rgb2hex(cc[j]), vmin=0, vmax=10, s=10, label=str(label))
        ax[i].set_title(md, fontsize=12)
        ax[i].legend()
    plt.tight_layout()
    return plt

import time
import datetime
import pandas as pd
from tqdm import tqdm
import rdkit.Chem as Chem
from sklearn.manifold import TSNE
from mordred import Calculator, descriptors


SELECTED_MD = ['nN', 'nP', 'nS', 'nO', 'nX', 'nBondsD', 'nBondsT', 'naRing', 'nARing']


def cal_md_by_smiles(smiles_list, md_list=None, print_info=False,
                     desc=(descriptors.AtomCount, descriptors.BondCount, descriptors.RingCount)):
    """
    calculate molecular descriptors by Mordred, https://github.com/mordred-descriptor/mordred
    :param smiles_list: list
            a list of smiles
    :param md_list: list
            a list of MD that need to calculate
    :param print_info: bool
    :param desc: tuple
            descriptors in mordred which contains MD we want to calculate
            common desc:  descriptors.AtomCount, descriptors.BondCount, descriptors.RingCount,
                          descriptors.Weight, descriptors.RotatableBond, descriptors.SLogP
    :return: pandas.DataFrame
    """
    # SELECTED_MD = ['nN', 'nS', 'nO', 'nX', 'nBondsD', 'nBondsT', 'naRing', 'nARing']
    if print_info:
        print('  >There are {} SMILES in this list'.format(len(smiles_list)))
        print('  >Top 3 SMILES: {}'.format(', '.join(smiles_list[0:3])))
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


def count_frag_in_mol_sentence(mol_sent_file_path):
    """
    count fragment in a molecular sentence file, fragments separated by space
    example: 2976033787 675765711 2976033787 675765711 2976033787 675765711 2976033787 954800030 2976033787 675765711
             2976033787 675765711 864662311 266675433 864662311 266675433 864674487 2324621955 984189120 3218370331
             864942730 2235918822 864662311 2192318254 864662311 2192318254 864662311 266675433 864662311
             266675433 864662311 266675433 (Mol2vec)
             or
             CN C1=CNC=NC1 C1=CNCCN1 CC CN N CN C1=CC=CC=C1 CC C CN CN C CC CC CC C C=O (parallel2vec)
    :param mol_sent_file_path: a molecular sentence file without header
           - one line one sentence
           - each fragment separated by space
    :return: a dataFrame contains fragment / n_fragment
    """
    frag2num = {}
    with open(mol_sent_file_path, 'r') as f_handle:
        for i in f_handle:
            i = i.strip()
            frags = i.split(' ')
            for frag in frags:
                if frag not in frag2num:
                    frag2num[frag] = 0
                frag2num[frag] += 1
    frag2num_df = pd.DataFrame.from_dict(data=frag2num, orient='index', columns=['n_fragment'])
    return frag2num_df


def _read_corpus(file_name):
    while True:
        line = file_name.readline()
        if not line:
            break
        yield line.split()


def insert_unk(corpus, out_corpus, threshold=3, uncommon='UNK', need_replace_list=None):
    """
    Handling of uncommon "words" (i.e. identifiers).
    It finds all least common identifiers (defined by threshold) and replaces them by 'uncommon' string.

    Parameters
    ----------
    corpus : str
        Input corpus file
    out_corpus : str
        Outfile corpus file
    threshold : int
        Number of identifier occurrences to consider it uncommon, < this threshold
    uncommon : str
        String to use to replace uncommon words/identifiers
    need_replace_list: list
        replace "words" in this list by uncommon

    Returns
    -------
    """
    if need_replace_list is None:
        need_replace_list = []
    # Find least common identifiers in corpus
    f = open(corpus)
    unique = {}
    i = 0
    for i, x in tqdm(enumerate(_read_corpus(f)), desc='Counting identifiers in corpus'):
        for identifier in x:
            if identifier not in unique:
                unique[identifier] = 1
            else:
                unique[identifier] += 1
    n_lines = i + 1
    least_common = set([x for x in unique if unique[x] < threshold])
    f.close()

    f = open(corpus)
    fw = open(out_corpus, mode='w')
    if len(need_replace_list) >= 1:
        least_common.update(need_replace_list)
    for line in tqdm(_read_corpus(f), total=n_lines, desc='Inserting %s' % uncommon):
        intersection = set(line) & least_common
        if len(intersection) > 0:
            new_line = []
            for item in line:
                if item in least_common:
                    new_line.append(uncommon)
                else:
                    new_line.append(item)
            fw.write(" ".join(new_line) + '\n')
        else:
            fw.write(" ".join(line) + '\n')
    f.close()
    fw.close()


def reduce_by_tsne(x, n_jobs=4):
    t0 = time.time()
    tsne = TSNE(n_components=2, n_jobs=n_jobs, learning_rate=200,
                n_iter=2000, random_state=42, init='pca', verbose=1)
    X_reduced_tsne = tsne.fit_transform(x)
    # X_reduced_tsne = tsne.fit(x)
    print(X_reduced_tsne.shape)
    # np.save('X_reduced_tsne_pca_first', X_reduced_tsne2)
    t1 = time.time()
    print("t-SNE took {:.1f}s.".format(t1 - t0))
    return X_reduced_tsne
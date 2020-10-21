import io
import time
import requests
import datetime
import pandas as pd
from tqdm import tqdm
import rdkit.Chem as Chem
from itertools import zip_longest
from sklearn.manifold import TSNE
# from ..utility import PUBCHEM_BASE_URL
from mordred import Calculator, descriptors


PUBCHEM_BASE_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/{}/CSV'
SELECTED_MD = ['nN', 'nP', 'nS', 'nO', 'nX', 'nBondsD', 'nBondsT', 'naRing', 'nARing']
MD_IMPORTANCE = {'nARing': 0, 'naRing': 1, 'nBondsT': 2, 'nBondsD': 3, 'nO': 4, 'nN': 5, 'nP': 6, 'nS': 7, 'nX': 8}


def cal_md_by_smiles_file(smiles_file_path, n=100000):
    all_results = []
    all_cid2smiles = {}
    with open(smiles_file_path) as f:
        for lines in grouper(f, n):
            assert len(lines) == n
            # process N lines here
            current_cid2smiles = {}
            for line in lines:
                if ',' in line:
                    line = line.strip().split(',')
                else:
                    line = line.strip().split('\t')
                cid, smiles = line
                all_cid2smiles[cid] = smiles
                current_cid2smiles[cid] = smiles
            current_smiles_list = list(current_cid2smiles.values())
            all_results.append(cal_md_by_smiles(current_smiles_list))
    all_results_df = pd.concat(all_results)
    all_results_df.index = all_results_df.index.map()


def cal_md_by_smiles(smiles_list, md_list=None, print_info=False, molecule_md=False,
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
        md_list = get_ordered_md()
    if print_info:
        print_df(md_df)
        # print('>>> The length of smiles_list: {}'.format(len(smiles_list)))
    md_df['smiles'] = smiles_list
    md_df = md_df.loc[:, ['smiles'] + md_list].copy()
    if print_info:
        print('  >The shape of smiles_info is: {}'.format(md_df.shape))
    if not molecule_md:
        md_df.rename(columns={'smiles': 'fragment'}, inplace=True)
        md_df.set_index('fragment', inplace=True)
    else:
        md_df.set_index('smiles', inplace=True)
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


def get_ordered_md():
    """
    get ordered Molecular descriptors according to MD_IMPORTANCE
    :return:
    """
    global MD_IMPORTANCE
    order2md = {i: md for md, i in MD_IMPORTANCE.items()}
    ordered_md = [order2md[i] for i in range(len(MD_IMPORTANCE))]
    return ordered_md


def get_mol_vec(frag2vec_file_path, data_set, result_path):
    """
    sum all fragment vector to get molecule vector
    :param frag2vec_file_path:
    :param data_set: the file path of selected molecules
        contains cid and fragments(id or SMILES),
        for example: 15054491	CO,CO,CC,CO,CO,C1=CC=CC=C1,C1CCOC1
    :param result_path:
    :return:
    """
    frag2vec_df = pd.read_csv(frag2vec_file_path, index_col=0)
    cid2vec = {}
    counter = 0
    print('>>> frag2vec')
    print_df(frag2vec_df)
    cid_coll = {}
    with open(data_set, 'r') as f_handle:
        for row in tqdm(f_handle):
            if not row.startswith('cid'):
                cid, frag_smiles = row.strip().split('\t')  # cid/frag_smiles (or frag_ids)
                cid_coll[cid] = 1
                frags = frag_smiles.split(',')
                # replace rare fragments (occurs in <5 molecules in the whole training set) to UNK
                frags = [i if i in frag2vec_df.index else 'UNK' for i in frags]
                cid2vec[cid] = frag2vec_df.loc[frags, :].sum().values
                if len(cid2vec) == 200000:
                    cid2vec_df = pd.DataFrame.from_dict(cid2vec, orient='index')
                    cid2vec_df.to_csv(result_path, mode='a', header=False, float_format='%.3f')
                    cid2vec = {}
            # if counter % 10000 == 0:
            #     print('>>> Processing line {}...'.format(counter))
            counter += 1
    # the last part
    # TODO: what's wrong in this piece of code???
    print('>>> the number of cid: {}'.format(len(cid_coll)))
    cid2vec_df = pd.DataFrame.from_dict(cid2vec, orient='index')
    cid2vec_df.to_csv(result_path, mode='a', header=False, float_format='%.3f')


def grouper(iterable, n, fillvalue=None):
    """
    https://stackoverflow.com/a/5845141/2803344
    read multiple lines each step, see example at function read_lines below
    :param iterable:
    :param n:
    :param fillvalue:
    :return:
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def read_lines(file_path, n=1000):
    """
    # https://stackoverflow.com/a/5845141/2803344
    :param file_path:
    :param n:
    :return:
    """
    with open(file_path) as f:
        for lines in grouper(f, n, ''):
            assert len(lines) == n
            # process N lines here


def query_smiles_by_cids(cid_list, mol_property='CanonicalSMILES'):
    """
    query SMILES by a CID list through URL
    only for a small set of CIDs
    :param cid_list: a list of CID
    :param mol_property:
    :return:
    """
    if len(cid_list) >= 200:
        cid_list = cid_list[:200]
    cids = ','.join([str(i) for i in cid_list])
    cid2smiles = {}
    result = requests.get(PUBCHEM_BASE_URL.format(cids, mol_property))
    if result.status_code == 200:
        # https://stackoverflow.com/a/32400969/2803344
        content = result.content
        c = pd.read_csv(io.StringIO(content.decode('utf-8')))
        # print(c.to_dict())
        for i in range(c.shape[0]):
            cid = c.loc[i, 'CID']
            prop_value = c.loc[i, mol_property]
            cid2smiles[cid] = prop_value
    return cid2smiles

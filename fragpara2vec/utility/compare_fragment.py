import re


def replace_nth(string, sub, wanted, n):
    # https://stackoverflow.com/a/35091558/2803344
    where = [m.start() for m in re.finditer(sub, string)][n-1]
    before = string[:where]
    after = string[where:]
    after = after.replace(sub, wanted, 1)
    new_string = before + after
    return new_string


def find_bond_pair(frag_df, bond_type):
    """
    find pairs of fragments which only is a single bond different
    such as: [(CC, C=C)], only need to plus a single double bond to get C=C from CC
    :param frag_df: a dataframe which contains molecular descriptors of each fragment
    :param bond_type: double_bond or triple_bond
    :return: list
    """
    assert bond_type in ['double_bond', 'triple_bond']
    all_frags = frag_df.index.to_list()
    # frag_df = frag_df.loc[frag_df['nARing'] <= 1].copy()
    paired_frag = {}
    bond_pairs = []
    if bond_type == 'double_bond':
        patten = '='
        frag_with_bonds = frag_df.loc[(frag_df['nBondsD'] >= 1) &
                                      (frag_df['nBondsD'] <= 3) &
                                      (frag_df['nBondsT'] == 0)]
    else:
        patten = '#'  # triple bond
        frag_with_bonds = frag_df.loc[(frag_df['nBondsT'] >= 1) &
                                      (frag_df['nBondsT'] <= 3) &
                                      (frag_df['nBondsD'] == 0)]

    for frag in frag_with_bonds.index:
        if frag not in paired_frag:
            n_bond = frag.count(patten)
            frag_remove_d_bond = []
            for i in range(n_bond):
                _frag_removed_d_bond = replace_nth(frag, patten, '', i+1)
                if (_frag_removed_d_bond in all_frags) and (_frag_removed_d_bond not in paired_frag):
                    paired_frag[_frag_removed_d_bond] = 1
                    paired_frag[frag] = 1
                    bond_pairs.append((frag, _frag_removed_d_bond))
                    break
    return bond_pairs


def find_aromatic_non_aroma_ring_pair(frag_df):
    """
    find pairs of fragments which is only different on aromatic bond
    eg: benzene ring(C1=CC=CC=C1) & hexahydrobenzene(C1CCCCC1)
    I remove all double bonds in aromatic ring to find corresponding non-aromatic ring molecule
    :param frag_df:
    :return:
    """
    all_frags = frag_df.index.to_list()
    # frag_df = frag_df.loc[frag_df['nARing'] <= 1].copy()
    paired_frag = {}
    bond_pairs = []

    # patten = 'naRing'
    frag_with_bonds = frag_df.loc[frag_df['naRing'] >= 1]

    for frag in frag_with_bonds.index:
        if frag not in paired_frag:
            _frag_removed_d_bond = frag.replace('=', '')
            if (_frag_removed_d_bond in all_frags) and (_frag_removed_d_bond not in paired_frag)\
                    and (frag != _frag_removed_d_bond):
                paired_frag[_frag_removed_d_bond] = 1
                paired_frag[frag] = 1
                bond_pairs.append((frag, _frag_removed_d_bond))
    return bond_pairs

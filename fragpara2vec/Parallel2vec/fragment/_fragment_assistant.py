import rdkit
import copy
import numpy as np
from typing import Dict
import rdkit.Chem as Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions


def decode_stereo(smiles2D):
    mol = Chem.MolFromSmiles(smiles2D)
    dec_isomers = list(EnumerateStereoisomers(mol))

    dec_isomers = [Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True)) for mol in dec_isomers]
    smiles3D = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in dec_isomers]

    chiralN = [atom.GetIdx() for atom in dec_isomers[0].GetAtoms() if
               int(atom.GetChiralTag()) > 0 and atom.GetSymbol() == "N"]
    if len(chiralN) > 0:
        for mol in dec_isomers:
            for idx in chiralN:
                mol.GetAtomWithIdx(idx).SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            smiles3D.append(Chem.MolToSmiles(mol, isomericSmiles=True))

    return smiles3D


def get_ring_and_merge(mol, common_atom_merge_ring=2):
    """
    get all simple rings in mol and merge two simple rings if they have >= common_atom_merge_ring common atoms
    :param mol:
    :param common_atom_merge_ring:
    :return:
    """
    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    id2ssr = {i: r_atom_ids for i, r_atom_ids in enumerate(ssr)}
    # Merge Rings with intersection >= 2 atoms
    while 1:
        id2ssr_dup = copy.deepcopy(id2ssr)
        for frag_id_a, atom_ids_a in id2ssr.items():
            for frag_id_b, atom_ids_b in id2ssr.items():
                if frag_id_a < frag_id_b:
                    inter = set(atom_ids_a) & set(atom_ids_b)
                    if len(inter) >= common_atom_merge_ring:
                        id2ssr_dup[frag_id_b].extend(id2ssr_dup[frag_id_a])  # merge two rings
                        id2ssr_dup[frag_id_b] = list(set(id2ssr_dup[frag_id_b]))
                        del id2ssr_dup[frag_id_a]  # remove fragment frag_id_b
        if id2ssr_dup == id2ssr:
            break
        id2ssr = copy.deepcopy(id2ssr_dup)  # update id2ssr and iteration again
    return [j for i, j in id2ssr.items()]


def get_clique_smiles(mol, atoms):
    """
    extract fragment SMILES by a list of atom index
    :param mol: Mol object in RDKit
    :param atoms: a list of atom index
    :return: the SMILES of fragment (atom id in atoms) in molecule
    """
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol)  # We assume this is not None
    mol_smiles = get_smiles(new_mol)
    return mol_smiles


def check_ending_fragment(n_atoms, frag_ids, frag2info):
    """
    check if there is any ending fragment in these fragments (>= 2)
    if == 1: and another fragment is ring, merge them together,
    if >= 2: merge all ending fragments,
    and update frag2atom
    else: return frag2atom without change
    :param n_atoms: the number of atoms
    :param frag_ids: a list which contains 3 fragment id
    :param frag2info: dict
        {2: FragInfo, ...} fragment with FragInfo class
    :return: frag2atom
    """
    frag_ids = sorted(frag_ids, reverse=True)  # big to small
    atom2frags = get_atom2frags(n_atoms=n_atoms, frag2info=frag2info)
    if_ring = np.array([frag2info[i].ring for i in frag_ids])
    if_end_fragment = np.array([np.any(np.array([atom2frags[i]]) == 1) for i in frag_ids])
    if np.any(if_end_fragment):  # at least 1 ending fragment exist
        count_non_ring_end_frag = 0  # non ring and ending fragment
        count_ring_non_end_frag = 0  # ring but not ending fragment
        for i, frag_id in enumerate(frag_ids):
            ring = if_ring[i]
            end_fragment = if_end_fragment[i]
            if ring and (not end_fragment):
                count_ring_non_end_frag += 1
            elif (not ring) and end_fragment:
                count_non_ring_end_frag += 1
        if len(frag_ids) == 2:
            if (count_ring_non_end_frag == 1) and (count_non_ring_end_frag == 1):
                # one ring with a non-ring end
                min_f, max_f = sorted(frag_ids)
                frag2info[min_f].atoms.extend(frag2info[max_f].atoms)
                frag2info[min_f].atoms = list(set(frag2info[min_f].atoms))
                del frag2info[max_f]
        elif sum(if_ring) == 0:
            # sum(if_ring) == count_ring_non_end_frag:
            # >=3 and no ring fragment
            if count_non_ring_end_frag == 2:  # only merge all ending fragments (>= 2)
                frag_ids = [i for i in frag_ids if if_end_fragment[i]]
                min_f = frag_ids.pop()
                for _frag_id in frag_ids:
                    frag2info[min_f].atoms.extend(frag2info[_frag_id].atoms)
                    del frag2info[_frag_id]
                frag2info[min_f].atoms = list(set(frag2info[min_f].atoms))

    return frag2info


def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol


def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


class FragInfo:
    def __init__(self, frag_id: int, atoms: list = None, smiles: str = '', ring: bool = False):
        self.frag_id = frag_id
        self.atoms = atoms
        self.smiles = smiles
        self.ring = ring

    def __str__(self):
        # https://stackoverflow.com/a/32635523/2803344
        return str(self.__class__) + ": " + str(self.__dict__)


def get_atom2frags(n_atoms, frag2info):
    """

    :param n_atoms:
    :param frag2info:
    :return:  {atom_id: [frag_id1, frag_id2], ...}
    """
    atom2frags = {i: [] for i in range(n_atoms)}  # {atom_id: [frag1, frag2], ...}
    for frag_id, frag_info in frag2info.items():
        for atom in frag_info.atoms:
            atom2frags[atom].append(frag_id)
    return atom2frags

import rdkit
import rdkit.Chem as Chem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict

from .vocab import Vocab
from ._fragment_assistant import (get_ring_and_merge, copy_edit_mol,
                                  copy_atom, sanitize, get_smiles,
                                  get_clique_smiles, FragInfo,
                                  get_atom2frags, check_ending_fragment)

MST_MAX_WEIGHT = 100
MAX_NCAND = 2000


def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)


def get_edges(n_atoms, frag2info, refragment=False):
    """
    get the relation between each two adjacent fragments
    :param n_atoms:
    :param frag2info: fragment id with FragInfo class
    :param refragment: whether do refragment,
        algorithm 1 in my paper
        change the rules in original paper for fragmentation, in function check_ending_fragment
    :return:
    """
    # Build edges and add singleton cliques
    # the edge is the relation between two fragments
    atom2frags = get_atom2frags(n_atoms=n_atoms, frag2info=frag2info)
    edges = defaultdict(int)  # https://stackoverflow.com/a/5900628/2803344
    for atom_inx in range(n_atoms):
        frag_inx = atom2frags[atom_inx]
        if len(frag_inx) >= 2:  # ignore the atoms only appear in one fragment
            if refragment:  # re-fragment
                frag2info = check_ending_fragment(n_atoms=n_atoms, frag_ids=frag_inx, frag2info=frag2info)
                atom2frags = get_atom2frags(n_atoms=n_atoms, frag2info=frag2info)
                frag_inx = atom2frags[atom_inx]
            if len(frag_inx) >= 2:
                # non-ring fragment contains two atoms, such as [1, 2]
                bonds = [c for c in frag_inx if (not frag2info[c].ring)]
                rings = [c for c in frag_inx if frag2info[c].ring]  # > 4 atoms in the fragment
                # In general, if len(frag_inx) >= 3, a singleton should be added,
                # but 1 bond + 2 ring is currently not dealt with.

                if len(bonds) > 2 or (len(bonds) == 2 and len(frag_inx) > 2):
                    # frag2atom[len(frag2atom)] = [atom_inx]
                    # frag2info[len(frag2info)].atoms = [atom_inx]
                    frag_id = len(frag2info)  # singleton clique which appear in more than 3 fragments
                    frag2info[frag_id] = FragInfo(frag_id=frag_id, atoms=[atom_inx])
                    c2 = len(frag2info) - 1  # fragment index for new singleton
                    for c1 in frag_inx:
                        edges[(c1, c2)] = 1

                elif len(rings) > 2:  # appear in multiple complex rings
                    # frag2atom[len(frag2atom)] = [atom_inx]
                    frag_id = len(frag2info)
                    frag2info[frag_id] = FragInfo(frag_id=frag_id, atoms=[atom_inx])
                    # frag2info[len(frag2info)].atoms = [atom_inx]
                    c2 = len(frag2info) - 1
                    for c1 in frag_inx:
                        edges[(c1, c2)] = MST_MAX_WEIGHT - 1
                else:
                    for i in range(len(frag_inx)):
                        for j in range(i + 1, len(frag_inx)):
                            c1, c2 = frag_inx[i], frag_inx[j]
                            inter = set(frag2info[c1].atoms) & set(frag2info[c2].atoms)
                            if edges[(c1, c2)] < len(inter):
                                edges[(c1, c2)] = len(inter)  # frag_inx[i] < frag_inx[j] by construction
    # [(0, 17, 99), (1, 17, 99), ...] which contain (fragment id, fragment id, weight)
    edges = [u + (MST_MAX_WEIGHT - v,) for u, v in edges.items()]
    if len(edges) == 0:
        return edges, frag2info

    # Compute Maximum Spanning Tree
    row, col, data = zip(*edges)
    n_clique = len(frag2info)
    clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
    junc_tree = minimum_spanning_tree(clique_graph)
    row, col = junc_tree.nonzero()
    edges = [(row[i], col[i]) for i in range(len(row))]
    frag2info = {i: j for i, j in frag2info.items() if len(j.atoms) != 0}
    return edges, frag2info  # fragments and the relation between each two fragments


def tree_decomp(mol, common_atom_merge_ring=2, refragment=False):
    """

    :param mol: <class 'rdkit.Chem.rdchem.Mol'> in RDKit
    :param common_atom_merge_ring: 2/3, if the intersection atom between two rings >= this number, merge them
    :param refragment:
    :return:
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:  # special case
        return [[0]], []

    # frag2atom = {}  # {0: [0, 1], 1: [1, 2], ...}, fragment id with a list of atom id
    frag2info = {}  # {fragment_id: FragInfo class, ...}
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            frag_id = len(frag2info)
            frag2info[frag_id] = FragInfo(frag_id=frag_id, atoms=[a1, a2])
            # frag2atom[len(frag2atom)] = [a1, a2]

    # ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    ssr = get_ring_and_merge(mol=mol,
                             common_atom_merge_ring=common_atom_merge_ring)
    for _ssr in ssr:
        frag_id = len(frag2info)
        frag2info[frag_id] = FragInfo(frag_id=frag_id, atoms=_ssr, ring=True)
        # frag2atom[len(frag2atom)] = _ssr

    edges, frag2info = get_edges(n_atoms=n_atoms, frag2info=frag2info, refragment=refragment)
    # frag2atom = {i: j.atoms for i, j in frag2info.items()}
    return edges, frag2info


def atom_equal(a1, a2):
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()


# Bond type not considered because all aromatic (so SINGLE matches DOUBLE)
def ring_bond_equal(b1, b2, reverse=False):
    b1 = (b1.GetBeginAtom(), b1.GetEndAtom())
    if reverse:
        b2 = (b2.GetEndAtom(), b2.GetBeginAtom())
    else:
        b2 = (b2.GetBeginAtom(), b2.GetEndAtom())
    return atom_equal(b1[0], b2[0]) and atom_equal(b1[1], b2[1])


def attach_mols(ctr_mol, neighbors, prev_nodes, nei_amap):
    prev_nids = [node.nid for node in prev_nodes]
    for nei_node in prev_nodes + neighbors:
        nei_id, nei_mol = nei_node.nid, nei_node.mol
        amap = nei_amap[nei_id]
        for atom in nei_mol.GetAtoms():
            if atom.GetIdx() not in amap:
                new_atom = copy_atom(atom)
                amap[atom.GetIdx()] = ctr_mol.AddAtom(new_atom)

        if nei_mol.GetNumBonds() == 0:
            nei_atom = nei_mol.GetAtomWithIdx(0)
            ctr_atom = ctr_mol.GetAtomWithIdx(amap[0])
            ctr_atom.SetAtomMapNum(nei_atom.GetAtomMapNum())
        else:
            for bond in nei_mol.GetBonds():
                a1 = amap[bond.GetBeginAtom().GetIdx()]
                a2 = amap[bond.GetEndAtom().GetIdx()]
                if ctr_mol.GetBondBetweenAtoms(a1, a2) is None:
                    ctr_mol.AddBond(a1, a2, bond.GetBondType())
                elif nei_id in prev_nids:  # father node overrides
                    ctr_mol.RemoveBond(a1, a2)
                    ctr_mol.AddBond(a1, a2, bond.GetBondType())
    return ctr_mol


def local_attach(ctr_mol, neighbors, prev_nodes, amap_list):
    ctr_mol = copy_edit_mol(ctr_mol)
    nei_amap = {nei.nid: {} for nei in prev_nodes + neighbors}

    for nei_id, ctr_atom, nei_atom in amap_list:
        nei_amap[nei_id][nei_atom] = ctr_atom

    ctr_mol = attach_mols(ctr_mol, neighbors, prev_nodes, nei_amap)
    return ctr_mol.GetMol()


# This version records idx mapping between ctr_mol and nei_mol
def enum_attach(ctr_mol, nei_node, amap, singletons):
    nei_mol, nei_idx = nei_node.mol, nei_node.nid
    att_confs = []
    black_list = [atom_idx for nei_id, atom_idx, _ in amap if nei_id in singletons]
    ctr_atoms = [atom for atom in ctr_mol.GetAtoms() if atom.GetIdx() not in black_list]
    ctr_bonds = [bond for bond in ctr_mol.GetBonds()]

    if nei_mol.GetNumBonds() == 0:  # neighbor singleton
        nei_atom = nei_mol.GetAtomWithIdx(0)
        used_list = [atom_idx for _, atom_idx, _ in amap]
        for atom in ctr_atoms:
            if atom_equal(atom, nei_atom) and atom.GetIdx() not in used_list:
                new_amap = amap + [(nei_idx, atom.GetIdx(), 0)]
                att_confs.append(new_amap)

    elif nei_mol.GetNumBonds() == 1:  # neighbor is a bond
        bond = nei_mol.GetBondWithIdx(0)
        bond_val = int(bond.GetBondTypeAsDouble())
        b1, b2 = bond.GetBeginAtom(), bond.GetEndAtom()

        for atom in ctr_atoms:
            # Optimize if atom is carbon (other atoms may change valence)
            if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() < bond_val:
                continue
            if atom_equal(atom, b1):
                new_amap = amap + [(nei_idx, atom.GetIdx(), b1.GetIdx())]
                att_confs.append(new_amap)
            elif atom_equal(atom, b2):
                new_amap = amap + [(nei_idx, atom.GetIdx(), b2.GetIdx())]
                att_confs.append(new_amap)
    else:
        # intersection is an atom
        for a1 in ctr_atoms:
            for a2 in nei_mol.GetAtoms():
                if atom_equal(a1, a2):
                    # Optimize if atom is carbon (other atoms may change valence)
                    if a1.GetAtomicNum() == 6 and a1.GetTotalNumHs() + a2.GetTotalNumHs() < 4:
                        continue
                    new_amap = amap + [(nei_idx, a1.GetIdx(), a2.GetIdx())]
                    att_confs.append(new_amap)

        # intersection is an bond
        if ctr_mol.GetNumBonds() > 1:
            for b1 in ctr_bonds:
                for b2 in nei_mol.GetBonds():
                    if ring_bond_equal(b1, b2):
                        new_amap = amap + [(nei_idx, b1.GetBeginAtom().GetIdx(), b2.GetBeginAtom().GetIdx()),
                                           (nei_idx, b1.GetEndAtom().GetIdx(), b2.GetEndAtom().GetIdx())]
                        att_confs.append(new_amap)

                    if ring_bond_equal(b1, b2, reverse=True):
                        new_amap = amap + [(nei_idx, b1.GetBeginAtom().GetIdx(), b2.GetEndAtom().GetIdx()),
                                           (nei_idx, b1.GetEndAtom().GetIdx(), b2.GetBeginAtom().GetIdx())]
                        att_confs.append(new_amap)
    return att_confs


# Try rings first: Speed-Up
def enum_assemble(node, neighbors, prev_nodes=[], prev_amap=[]):
    all_attach_confs = []
    singletons = [nei_node.nid for nei_node in neighbors + prev_nodes if nei_node.mol.GetNumAtoms() == 1]

    def search(cur_amap, depth):
        if len(all_attach_confs) > MAX_NCAND:
            return
        if depth == len(neighbors):
            all_attach_confs.append(cur_amap)
            return

        nei_node = neighbors[depth]
        cand_amap = enum_attach(node.mol, nei_node, cur_amap, singletons)
        cand_smiles = set()
        candidates = []
        for amap in cand_amap:
            cand_mol = local_attach(node.mol, neighbors[:depth + 1], prev_nodes, amap)
            cand_mol = sanitize(cand_mol)
            if cand_mol is None:
                continue
            smiles = get_smiles(cand_mol)
            if smiles in cand_smiles:
                continue
            cand_smiles.add(smiles)
            candidates.append(amap)

        if len(candidates) == 0:
            return

        for new_amap in candidates:
            search(new_amap, depth + 1)

    search(prev_amap, 0)
    cand_smiles = set()
    candidates = []
    aroma_score = []
    for amap in all_attach_confs:
        cand_mol = local_attach(node.mol, neighbors, prev_nodes, amap)
        cand_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cand_mol))
        smiles = Chem.MolToSmiles(cand_mol)
        if smiles in cand_smiles or check_singleton(cand_mol, node, neighbors) == False:
            continue
        cand_smiles.add(smiles)
        candidates.append((smiles, amap))
        aroma_score.append(check_aroma(cand_mol, node, neighbors))

    return candidates, aroma_score


def check_singleton(cand_mol, ctr_node, nei_nodes):
    rings = [node for node in nei_nodes + [ctr_node] if node.mol.GetNumAtoms() > 2]
    singletons = [node for node in nei_nodes + [ctr_node] if node.mol.GetNumAtoms() == 1]
    if len(singletons) > 0 or len(rings) == 0: return True

    n_leaf2_atoms = 0
    for atom in cand_mol.GetAtoms():
        nei_leaf_atoms = [a for a in atom.GetNeighbors() if not a.IsInRing()]  # a.GetDegree() == 1]
        if len(nei_leaf_atoms) > 1:
            n_leaf2_atoms += 1

    return n_leaf2_atoms == 0


def check_aroma(cand_mol, ctr_node, nei_nodes):
    rings = [node for node in nei_nodes + [ctr_node] if node.mol.GetNumAtoms() >= 3]
    if len(rings) < 2: return 0  # Only multi-ring system needs to be checked

    get_nid = lambda x: 0 if x.is_leaf else x.nid
    benzynes = [get_nid(node) for node in nei_nodes + [ctr_node] if node.smiles in Vocab.benzynes]
    penzynes = [get_nid(node) for node in nei_nodes + [ctr_node] if node.smiles in Vocab.penzynes]
    if len(benzynes) + len(penzynes) == 0:
        return 0  # No specific aromatic rings

    n_aroma_atoms = 0
    for atom in cand_mol.GetAtoms():
        if atom.GetAtomMapNum() in benzynes + penzynes and atom.GetIsAromatic():
            n_aroma_atoms += 1

    if n_aroma_atoms >= len(benzynes) * 4 + len(penzynes) * 3:
        return 1000
    else:
        return -0.001

#     # Only used for debugging purpose


def dfs_assemble(cur_mol, global_amap, fa_amap, cur_node, fa_node):
    fa_nid = fa_node.nid if fa_node is not None else -1
    prev_nodes = [fa_node] if fa_node is not None else []

    children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
    neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
    neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
    singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
    neighbors = singletons + neighbors

    cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in fa_amap if nid == cur_node.nid]
    cands = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)

    cand_smiles, cand_amap = zip(*cands)
    label_idx = cand_smiles.index(cur_node.label)
    label_amap = cand_amap[label_idx]

    for nei_id, ctr_atom, nei_atom in label_amap:
        if nei_id == fa_nid:
            continue
        global_amap[nei_id][nei_atom] = global_amap[cur_node.nid][ctr_atom]

    cur_mol = attach_mols(cur_mol, children, [], global_amap)  # father is already attached
    for nei_node in children:
        if not nei_node.is_leaf:
            dfs_assemble(cur_mol, global_amap, label_amap, nei_node, cur_node)


if __name__ == "__main__":
    pass
    # import sys
    # from fast_jtnn.mol_tree import MolTree
    #
    # lg = rdkit.RDLogger.logger()
    # lg.setLevel(rdkit.RDLogger.CRITICAL)
    #
    # smiles = ["O=C1[C@@H]2C=C[C@@H](C=CC2)C1(c1ccccc1)c1ccccc1", "O=C([O-])CC[C@@]12CCCC[C@]1(O)OC(=O)CC2",
    #           "ON=C1C[C@H]2CC3(C[C@@H](C1)c1ccccc12)OCCO3",
    #           "C[C@H]1CC(=O)[C@H]2[C@@]3(O)C(=O)c4cccc(O)c4[C@@H]4O[C@@]43[C@@H](O)C[C@]2(O)C1",
    #           'Cc1cc(NC(=O)CSc2nnc3c4ccccc4n(C)c3n2)ccc1Br', 'CC(C)(C)c1ccc(C(=O)N[C@H]2CCN3CCCc4cccc2c43)cc1',
    #           "O=c1c2ccc3c(=O)n(-c4nccs4)c(=O)c4ccc(c(=O)n1-c1nccs1)c2c34", "O=C(N1CCc2c(F)ccc(F)c2C1)C1(O)Cc2ccccc2C1"]
    #
    #
    # def tree_test():
    #     for s in sys.stdin:
    #         s = s.split()[0]
    #         tree = MolTree(s)
    #         print('-------------------------------------------')
    #         print(s)
    #         for node in tree.nodes:
    #             print(node.smiles, [x.smiles for x in node.neighbors])
    #
    #
    # def decode_test():
    #     wrong = 0
    #     for tot, s in enumerate(sys.stdin):
    #         s = s.split()[0]
    #         tree = MolTree(s)
    #         tree.recover()
    #
    #         cur_mol = copy_edit_mol(tree.nodes[0].mol)
    #         global_amap = [{}] + [{} for node in tree.nodes]
    #         global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}
    #
    #         dfs_assemble(cur_mol, global_amap, [], tree.nodes[0], None)
    #
    #         cur_mol = cur_mol.GetMol()
    #         cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
    #         set_atommap(cur_mol)
    #         dec_smiles = Chem.MolToSmiles(cur_mol)
    #
    #         gold_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(s))
    #         if gold_smiles != dec_smiles:
    #             print(gold_smiles, dec_smiles)
    #             wrong += 1
    #         print(wrong, tot + 1)
    #
    #
    # def enum_test():
    #     for s in sys.stdin:
    #         s = s.split()[0]
    #         tree = MolTree(s)
    #         tree.recover()
    #         tree.assemble()
    #         for node in tree.nodes:
    #             if node.label not in node.cands:
    #                 print(tree.smiles)
    #                 print(node.smiles, [x.smiles for x in node.neighbors])
    #                 print(node.label, len(node.cands))
    #
    #
    # def count():
    #     cnt, n = 0, 0
    #     for s in sys.stdin:
    #         s = s.split()[0]
    #         tree = MolTree(s)
    #         tree.recover()
    #         tree.assemble()
    #         for node in tree.nodes:
    #             cnt += len(node.cands)
    #         n += len(tree.nodes)
    #         # print cnt * 1.0 / n
    #
    #
    # count()

"""
The first step: generate molecular tree based on https://github.com/wengong-jin/icml18-jtnn
Then we can get the fragments of each molecule and the relation between each two fragments

# usage:

$ python ./fragment/s1_mol_tree.py
 ./demo_data/02_filtered_molecule/cid2SMILES_after_filtered.txt
 ./demo_data/03_fragment/mol_tree.txt --log_fn
 ./demo_data/03_fragment/mol_tree.log
"""
import os
import json
import rdkit
import argparse
from tqdm import tqdm
import rdkit.Chem as Chem
from .chemutils import tree_decomp, set_atommap, enum_assemble
from .pub_func import write_list_by_json
from ._fragment_assistant import get_clique_smiles, get_mol, get_smiles
# from .pub_func import read_json_line


class MolTreeNode(object):

    def __init__(self, frag_smiles, clique=[]):
        self.smiles = frag_smiles
        self.mol = get_mol(frag_smiles)

        self.clique = [x for x in clique]  # copy
        self.neighbors = []

    def add_neighbor(self, nei_node):
        self.neighbors.append(nei_node)

    # def recover(self, original_mol):
    #     clique = []
    #     clique.extend(self.clique)
    #     if not self.is_leaf:
    #         for cidx in self.clique:
    #             original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(self.nid)
    #
    #     for nei_node in self.neighbors:
    #         clique.extend(nei_node.clique)
    #         if nei_node.is_leaf:  # Leaf node, no need to mark
    #             continue
    #         for cidx in nei_node.clique:
    #             # allow singleton node override the atom mapping
    #             if cidx not in self.clique or len(nei_node.clique) == 1:
    #                 atom = original_mol.GetAtomWithIdx(cidx)
    #                 atom.SetAtomMapNum(nei_node.nid)
    #
    #     clique = list(set(clique))
    #     label_mol = get_clique_smiles(original_mol, clique)
    #     self.label = Chem.MolToSmiles(Chem.MolFromSmiles(label_mol))
    #
    #     for cidx in clique:
    #         original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)
    #
    #     return self.label
    #
    # def assemble(self):
    #     neighbors = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() > 1]
    #     neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
    #     singletons = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() == 1]
    #     neighbors = singletons + neighbors
    #
    #     cands, aroma = enum_assemble(self, neighbors)
    #     new_cands = [cand for i, cand in enumerate(cands) if aroma[i] >= 0]
    #     if len(new_cands) > 0:
    #         cands = new_cands
    #
    #     if len(cands) > 0:
    #         self.cands, _ = zip(*cands)
    #         self.cands = list(self.cands)
    #     else:
    #         self.cands = []


class MolTree(object):

    def __init__(self, smiles, common_atom_merge_ring=3, refragment=False):
        """

        :param smiles:
        :param common_atom_merge_ring: common_atom_merge_ring: 2/3,
            if the intersection atom between two rings >= this number, merge them
        """
        self.smiles = smiles
        self.mol = get_mol(smiles)

        # Stereo Generation (currently disabled)
        # mol = Chem.MolFromSmiles(smiles)
        # self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        # self.smiles2D = Chem.MolToSmiles(mol)
        # self.stereo_cands = decode_stereo(self.smiles2D)

        edges, frag2info = tree_decomp(self.mol, common_atom_merge_ring, refragment=refragment)

        # add fragment SMILES information
        for frag_id, frag_info in frag2info.items():
            frag2info[frag_id].smiles = get_clique_smiles(self.mol, frag_info.atoms)

        for x, y in edges:
            frag2info[x].append(y)
            frag2info[y].append(x)

        # self.nodes = {}  # node is fragment in graph
        # root = 0
        # for i, atom_info in frag2info.items():  # atoms is a list of atom id
        #     # frag_smiles = get_clique_smiles(self.mol, atoms)
        #     node = MolTreeNode(frag_smiles=atom_info.smiles, clique=atom_info.atoms)
        #     self.nodes[i] = node  #
        #     if min(atom_info.atoms) == 0:
        #         root = i
        # a = 3 + 4
        # for x, y in edges:
        #     self.nodes[x].add_neighbor(self.nodes[y])
        #     self.nodes[y].add_neighbor(self.nodes[x])

        # if root > 0:
        #     self.nodes[0], self.nodes[root] = self.nodes[root], self.nodes[0]

        # for i, node in self.nodes.items():
        #     node.nid = i
        #     if len(node.neighbors) > 1:  # Leaf node mol is not marked
        #         set_atommap(node.mol, node.nid)
        #     node.is_leaf = (len(node.neighbors) == 1)

    # def size(self):
    #     return len(self.nodes)

    # def recover(self):
    #     for node in self.nodes:
    #         node.recover(self.mol)
    #
    # def assemble(self):
    #     for node in self.nodes:
    #         node.assemble()


def dfs(node, fa_idx):
    max_depth = 0
    for child in node.neighbors:
        if child.idx == fa_idx:
            continue
        max_depth = max(max_depth, dfs(child, node.idx))
    return max_depth + 1


def get_frag2info(mol_smiles, common_atom_merge_ring=3, refragment=False):
    """

    :param mol_smiles:
    :param common_atom_merge_ring:
    :param refragment:
    :return:
    """
    mol = get_mol(mol_smiles)
    edges, frag2info = tree_decomp(mol, common_atom_merge_ring, refragment=refragment)

    # # add fragment SMILES information
    # for frag_id, frag_info in frag2info.items():
    #     frag2info[frag_id].smiles = get_clique_smiles(mol, frag_info.atoms)
    #
    # for x, y in edges:
    #     frag2info[x].neighbors.append(int(y))
    #     frag2info[y].neighbors.append(int(x))

    return {'frag2info': frag2info, 'edges': edges}


def call_mol_tree(raw_data_file, result_dir, log_file, start_line=1,
                  only_fragment=False, common_atom_merge_ring=2,
                  clip=False, test_mode=False, ignore_existed_cid=True,
                  refragment=True, ref_cid_file_path=None):
    """
    molecule fragment by tree decomposition
    :param raw_data_file: training set file path,
        see file fragpara2vec/demo_data/02_filtered_molecule/cid2SMILES_after_filtered.txt
        cid and SMILES
    :param result_dir: result file dir
    :param log_file: log file name
    :param start_line: start from x line to do tree decomposition, used for debug
    :param only_fragment: only write file cid2frag_id2frag_smiles.txt
    :param common_atom_merge_ring: 2/3, if the intersection atom between two rings >= this number, merge them
    :param clip: if need to align all four result files and remove duplicates
    :param test_mode: if use test mode
    :param ignore_existed_cid:
    :param refragment: change the rules in original paper for fragmentation, in function check_ending_fragment
    :param ref_cid_file_path: only for clip, if this file is provided, clipping will use CID in this list to filter result
    :return: four files: cid2mol_smiles.txt, cid2frag_id2frag_smiles.txt, cid2frag_id2neighbors, cid2frag_id2mol_inx.txt
    """
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    # cset2count = {}  # count the number of fragment
    log_file_path = os.path.join(result_dir, log_file)
    columns = ['frag_id2mol_inx', 'edges', 'mol_smiles']
    if only_fragment:
        columns = ['frag_smiles']
    file_names = ['cid2' + i + '.txt' for i in columns]
    # result = pd.DataFrame()
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for frag in range(len(columns)):
        if not os.path.exists(os.path.join(result_dir, file_names[frag])):
            with open(os.path.join(result_dir, file_names[frag]), 'w') as f:
                f.write('\t'.join(['cid', columns[frag]]) + '\n')
    existed_cid = {}

    if os.path.exists(os.path.join(result_dir, file_names[-1])) and ignore_existed_cid:
        print('>>> read previous result (cid)...')
        with open(os.path.join(result_dir, file_names[-1]), 'r') as f:
            for frag in tqdm(f):
                if not frag.startswith('cid'):
                    cid, _ = frag.split('\t')
                    existed_cid[int(cid)] = 1
        if os.path.exists(log_file_path):
            print('>>> read log file (cid)...')
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('The number of fragment <= 3'):
                        cid = line.split('cid: ')[1]
                        existed_cid[int(cid)] = 1
        print('    there are {} molecules have been parsed'.format(len(existed_cid)))
    result_f_only_fragment = None
    if only_fragment:
        result_f_only_fragment = open(os.path.join(result_dir, file_names[0]), 'a')
    with open(raw_data_file) as f:
        counter = 1
        print('>>> start from line {}'.format(start_line))
        for line in tqdm(f):
            if counter >= int(start_line):
                # if counter % 5000 == 0:
                #     print('>>> current line: ', counter)
                node2edges = {}
                # frag_id2frag_smiles = {}  # frag_id2frag_smiles
                frag_id2atom_inx = {}  # atom index in molecule
                if ',' in line:
                    _line = line.strip().split(',')
                else:
                    _line = line.strip().split('\t')
                # print(_line)
                if len(_line) == 2:
                    cid, smiles = _line
                else:
                    raise Exception('Each line should separate by tab and 1 or 2 columns')
                # cid, smiles = line.strip().split('\t')
                if (cid.lower() != 'cid') and (int(cid) not in existed_cid):
                    if test_mode:
                        print('>>> current CID: {}'.format(cid))
                        print('>>> current SMILES: {}'.format(smiles))
                    mol_frag2info = {}
                    edges = []  # [(a1, a2), (b1, b2), ...]
                    mol = get_mol(smiles)
                    try:
                        edges, mol_frag2info = tree_decomp(mol, common_atom_merge_ring, refragment=refragment)
                        # mol_frag2info = {str(frag_id): frag_info for frag_id, frag_info in mol_frag2info.items()}
                        edges = [(str(i), str(j)) for i, j in edges]
                    except Exception as e:
                        with open(log_file_path, 'a') as log_f:
                            log_f.write('mol_tree error, cid: {}'.format(cid) + '\n')
                        # mol = MolTree(smiles, common_atom_merge_ring=common_atom_merge_ring, refragment=refragment)
                    if (len(edges) >= 3) and (len(mol_frag2info) > 3):
                        for frag_id, frag in mol_frag2info.items():
                            frag_id2atom_inx[frag_id] = frag.atoms
                        if only_fragment:
                            # add fragment SMILES information
                            frag_id2frag_smiles = {}
                            for frag_id, frag_info in mol_frag2info.items():
                                frag_id2frag_smiles[frag_id] = get_clique_smiles(mol, frag_info.atoms)
                            # col2val = {columns[0]: mol_blocks}
                            write_str = cid + '\t' + ','.join(list(frag_id2frag_smiles.values())) + '\n'
                            result_f_only_fragment.write(write_str)
                        else:
                            col2val = {columns[0]: frag_id2atom_inx,
                                       columns[1]: edges,
                                       columns[2]: smiles}
                            for j in range(len(columns)):
                                with open(os.path.join(result_dir, file_names[j]), 'a') as result_f:
                                    # print(columns[j])
                                    current_frag_info = col2val[columns[j]]
                                    write_str = '\t'.join([str(cid), json.dumps(current_frag_info)]) + '\n'
                                    # write_str = write_list_by_json([cid, col2val[columns[j]]])
                                    result_f.write(write_str)
                    else:
                        with open(log_file_path, 'a') as log_f:
                            log_f.write('The number of fragment <= 3, cid: {}'.format(cid) + '\n')
            counter += 1

        # sort result
        if not only_fragment and clip:
            new_file_names = ['cid2' + i + '_new.txt' for i in columns]
            for frag in range(len(columns)):
                with open(os.path.join(result_dir, new_file_names[frag]), 'w') as f:
                    f.write('\t'.join(['cid', columns[frag]]) + '\n')

            print('Start to clip all result files...')
            common_cid = _get_common_cid(columns=columns, file_names=file_names, result_dir=result_dir)
            if ref_cid_file_path is not None:
                cid_in_ref = {}
                print('>>> Read cid in ref file...')
                with open(ref_cid_file_path, 'r', encoding='utf-8') as f:
                    for line in tqdm(f):
                        cid, _ = line.split('\t')
                        cid_in_ref[cid] = 1
                # filter common_cid again by CID in ref
                _common_cid = {i: None for i in tqdm(common_cid)}
                common_cid = [i for i in cid_in_ref if i in _common_cid]
                del _common_cid
            print('>>> There are {} common cids in all result files'.format(len(common_cid)))
            print('>>> Write new ordered results...')
            for frag in range(len(columns)):
                current_info = {}
                with open(os.path.join(result_dir, file_names[frag]), 'r') as f_handle:
                    print('>>> Deal with file: {}'.format(file_names[frag]))
                    for line in f_handle:
                        if not line.startswith('cid'):
                            cid, frag_info = line.strip().split('\t')
                            current_info[cid] = frag_info
                with open(os.path.join(result_dir, new_file_names[frag]), 'a') as f_handle:
                    for cid in tqdm(common_cid):
                        # print('>>> Deal with file: {}'.format(new_file_names[i]))
                        f_handle.write('\t'.join([cid, current_info[cid]]) + '\n')
    if only_fragment:
        result_f_only_fragment.close()


def _get_common_cid(columns, file_names, result_dir):
    """

    :param file_names:
    :return: list
    """
    cid_in_each_file = {'cid2' + i: {} for i in columns}  # {cid2frag_id2mol_inx: {}, '': {}, ...}
    for i in range(len(columns)):
        current_info = 'cid2' + columns[i]
        with open(os.path.join(result_dir, file_names[i]), 'r') as f_handle:
            print('>>> Deal with file: {}'.format(file_names[i]))
            for line in tqdm(f_handle):
                if not line.startswith('cid'):
                    cid, _ = line.strip().split('\t')
                    if 'edges' == columns[i]:
                        edges = json.loads(_)
                        if len(edges) >= 3:  # at least 3 edges
                            cid_in_each_file[current_info][cid] = 0
                    else:
                        cid_in_each_file[current_info][cid] = 0
    cid_set = [list(j.keys()) for i, j in cid_in_each_file.items()]
    common_cid = list(set(cid_set[0]) & set(cid_set[1]) & set(cid_set[2]))
    return common_cid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='molecule fragment by tree decomposition')
    parser.add_argument('training_set_fn',
                        help='training set file path')
    parser.add_argument('result_fn',
                        help='result file path')
    parser.add_argument('--log_fn',
                        help='log file name')

    parser.add_argument('--start_line', default=1, help='start from this line')

    args = parser.parse_args()

    # root_dir = r'/home/belter/github/my-research/jtnn-py3'
    raw_data = args.training_set_fn
    # result_dir = args.result_fn


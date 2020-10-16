import datetime
import numpy as np
import pandas as pd
import networkx as nx
from rdkit import Chem
from itertools import product
from operator import itemgetter


def get_mol_obj(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
    except TypeError:
        m = Chem.MolFromSmarts(smiles)
    return m


def check_ring(smiles):
    """
    check if a SMILES is ring
    :param smiles:
    :return:
    """
    # https://github.com/rdkit/rdkit/issues/1984
    m = Chem.MolFromSmarts(smiles)
    m.UpdatePropertyCache()
    Chem.GetSymmSSSR(m)
    ring_info = m.GetRingInfo()
    if ring_info.NumRings() >= 1:
        return True
    else:
        return False


def get_num_atom_by_smiles(smiles):
    m = get_mol_obj(smiles)
    num_atom = m.GetNumAtoms()
    return num_atom


class Mol2Network:
    def __init__(self, smiles, n2n, id2smiles, id2mol_inx):
        """
        represent a molecule by graph
        :param smiles: the SMILES of this molecule, eg: C#CCN(CC#C)C(=O)c1cc2ccccc2cc1OC(F)F
        :param n2n: node to neighbors (node id and it's neighbors)
        - {1: [2], 2: [1, 3], 3: [2, 16], 4: [5, 16], 5: [4, 6], 6: [5],
           7: [16, 17], 8: [17], 9: [14, 17], 10: [11, 14], 11: [10, 18], 12: [18],
           13: [18], 14: [9, 10, 15], 15: [14], 16: [3, 4, 7], 17: [7, 8, 9], 18: [11, 12, 13]}
        :param id2smiles: fragment id 2 fragment smiles, eg:
        - {1: 'C#C', 2: 'CC', 3: 'CN', 4: 'CN', 5: 'CC', 6: 'C#C', 7: 'CN',
           8: 'C=O', 9: 'CC', 10: 'CO', 11: 'CO', 12: 'CF', 13: 'CF',
           14: 'C1=CCCC=C1', 15: 'C1=CC=CC=C1', 16: 'N', 17: 'C', 18: 'C'}
        :param id2mol_inx: fragment id to atom index in this molecule, eg:
        - {"1": [0, 1], "2": [1, 2], "3": [2, 3], "4": [3, 4], "5": [4, 5], "6": [5, 6], "7": [3, 7],
           "8": [7, 8], "9": [7, 9], "10": [18, 19], "11": [19, 20], "12": [20, 21], "13": [20, 22],
           "14": [9, 18, 17, 16, 11, 10], "15": [12, 13, 14, 15, 16, 11], "16": [3], "17": [7], "18": [20]}
        """
        # self.smiles = smiles
        self.n2n = n2n
        self.id2smiles = id2smiles
        self.id2mol_inx = id2mol_inx
        self.g = self._get_graph()
        self.end_points = self._get_end_points()

    def _get_graph(self):
        # network of molecule generated by networkx
        g = nx.Graph()
        for i in self.n2n.keys():
            edges = [(i, j) for j in self.n2n[i]]
            g.add_edges_from(edges)

        id2smile_attr = {k: {'smiles': v} for k, v in self.id2smiles.items()}
        id2mol_inx_attr = {int(k): {'mol_inx': v} for k, v in self.id2mol_inx.items()}
        nx.set_node_attributes(g, id2smile_attr)
        nx.set_node_attributes(g, id2mol_inx_attr)
        return g

    def _get_end_points(self):
        end_points = [i for i in self.n2n if len(self.n2n[i]) == 1]
        return end_points

    def _get_end_pairs(self):
        # get end point (only one neighbor) pairs (all combination)
        end_points = self.end_points
        num_ = len(end_points)
        end_pairs = []
        if num_ > 2:
            for i in range(num_):
                for j in range(num_):
                    if i < j:
                        end_pairs.append((end_points[i], end_points[j]))
        else:
            end_pairs.append(tuple(end_points))
        return end_pairs

    def count_neighbors(self, mol_path):
        num_neighbors = 0
        for i in mol_path:
            _n_n = len(self.n2n[i])
            num_neighbors += _n_n
        return num_neighbors

    def get_mol_path(self):
        """
        get all molecular paths from end to end (end pairs of atom in molecule)
        :return:
        """
        end_pairs = self._get_end_pairs()  # get end point pairs
        paths_with_attr = []  # with attribute
        all_paths = []
        num_node_longest_path = 0
        num_max_path_neighbors = 0
        # num_max_atom_in_path = 0
        if len(self.n2n) >= 2:
            for pairs in end_pairs:
                # print(pairs)
                shortest_path = nx.shortest_simple_paths(self.g, pairs[0], pairs[1])
                shortest_path = list(shortest_path)[0]
                num_neig = self.count_neighbors(shortest_path)
                num_atoms = np.sum([get_num_atom_by_smiles(self.id2smiles[i]) for i in shortest_path])
                paths_with_attr.append({'path': shortest_path, 'len_path': len(shortest_path),
                                        'num_neig': num_neig, 'num_atoms': num_atoms})
                # print(shortest_path)
                if len(shortest_path) > num_node_longest_path:
                    num_node_longest_path = len(shortest_path)
                if num_neig > num_max_path_neighbors:
                    num_max_path_neighbors = num_neig
                # if num_atoms > num_max_atom_in_path:
                #     num_max_atom_in_path = num_atoms
            paths_with_attr = sorted(paths_with_attr, key=itemgetter('num_atoms'), reverse=True)
            paths_with_attr = sorted(paths_with_attr, key=itemgetter('num_neig'), reverse=True)  # test data LINE 401
            for path in paths_with_attr:
                # print('>>>>>>>>>> this is path <<<<<<<<<')
                # print(path)
                all_paths.append(path['path'])
        if len(self.n2n) == 1:  # only one node in all graph, test data LINE 546
            all_paths = list(self.n2n.keys())
        return all_paths

    def get_id2node(self, mol_path, id2smile, id2mol_inx):
        """
        fragment id maps to Node class
        :param mol_path: mol-tree id in the longest path (skeleton), a list
        :param id2smile: id to SMILES
        :param id2mol_inx: id to index in molecule
        :return: {id: Node class, ...}
        """
        id2node = {}
        for i, smiles in id2smile.items():
            ring = check_ring(smiles)
            num_atom = get_num_atom_by_smiles(smiles)
            mol_inx = id2mol_inx[i]
            node_type = 'branch'
            branch_path = []
            if i in mol_path:
                node_type = 'skeleton'
                branch_path = self.get_skeleton_node_branch_path(mol_path, i)
            node = Node(tree_id=i, smiles=smiles, ring=ring, num_atom=num_atom,
                        mol_inx=mol_inx, node_type=node_type, branch_path=branch_path)
            id2node[i] = node
        for i, node in id2node.items():
            neis_tree_id = self.n2n[i]
            for nei in neis_tree_id:
                node.add_neighbor(id2node[nei])

        return id2node

    def get_skeleton_node_branch_path(self, mol_path, frag_id):
        """

        :param mol_path: all fragments on the skeleton
        :param frag_id: fragment id in mol tree
        :return: branch_path, a list of node id
        """
        direct_neighbors = self.n2n[frag_id]
        branch_path = []
        if len(direct_neighbors) >= 3:  # usually have 1 branch
            branch_neighbors = [i for i in direct_neighbors if i not in mol_path]
            # all_neighbors['skeleton_neighbors'] += [i for i in direct_neighbors if i in mol_path]
            for j in branch_neighbors:
                if j not in self.end_points:
                    for ep in self.end_points:
                        branch_path_tmp = nx.shortest_simple_paths(self.g, frag_id, ep)  # test data LINE 14
                        branch_path_tmp = list(branch_path_tmp)[0]
                        # remove frag_id, which is included in skeleton
                        branch_path_tmp = [node_id for node_id in branch_path_tmp if node_id != frag_id]
                        branch_path_tmp_bak = branch_path_tmp.copy()
                        if j in branch_path_tmp_bak:
                            # add other neighbors in this branch path
                            for branch_node in branch_path_tmp_bak:
                                branch_node_neighs = self.n2n[branch_node]
                                branch_node_neighs = [node_id for node_id in branch_node_neighs
                                                      if node_id not in mol_path]
                                for branch_node_neig in branch_node_neighs:
                                    if branch_node_neig not in branch_path_tmp_bak:
                                        branch_path_tmp.append(branch_node_neig)
                            branch_path.append(branch_path_tmp)
                            break
                else:
                    branch_path.append([j])  # only one node in this branch path

        return branch_path


class Node:
    def __init__(self, tree_id, smiles, ring,
                 num_atom, mol_inx, node_type, branch_path):
        self.id = tree_id
        self.smiles = smiles  # SMILES of this node (fragment/cluster)
        self.ring = ring  # True/False
        self.num_atom = num_atom  # int
        self.mol_inx = mol_inx  # a list
        self.neighbors = []  # all neighbors of this node
        self.type = node_type  # the type of this node, skeleton/branch
        self.branch_path = branch_path  # a list of node id

    def show_info(self):
        print('>>>>')
        print('id: ', self.id)
        print('smiles: ', self.smiles)
        print('num atom: ', self.num_atom)
        print('type: ', self.type)
        print('neighbors: ', self.neighbors)
        print('mol_inx: ', self.mol_inx)
        print()

    def add_neighbor(self, neighbor):
        # neighbor is a node class
        self.neighbors.append(neighbor)


def get_fragment_sentence(frag2num_fp, cid2frag_fp, result_fp, result_fp2, replace_by_id=True):
    """
    get fragment sentence
    replace SMILES of fragment by fragment id for saving storage space
    :param frag2num_fp: file path of fragment to number, frag_id,fragment(SMILES),count
    :param cid2frag_fp: file path of cid to sentence (fragments)
    :param result_fp: file path of cid to fragment id
    :param result_fp2: file path of fragment id sentence (separated by space)
    :param replace_by_id: ??
    :return:
    """
    frag2num = pd.read_csv(frag2num_fp, index_col='fragment')
    frag2id = frag2num['frag_id'].to_dict()  # a dict of mapping fragment SMILES to fragment id

    with open(cid2frag_fp, 'r') as f_handle:
        counter = 0
        for i in f_handle:
            if counter % 500000 == 0:
                t = datetime.datetime.now()
                print('>>> Current line: {}'.format(counter), t.strftime("%c"))
            cid, sentence = i.strip().split('\t')
            frags = sentence.split(',')
            if replace_by_id:  # replace fragment SMILES by id for saving storage
                frags_id = [str(frag2id[f]) for f in frags]
                result_fp = result_fp.replace('frag_smiles', 'frag_id')
                result_fp2 = result_fp2.replace('frag_smiles', 'frag_id')
                with open(result_fp, 'a') as f_handle2:
                    f_handle2.write(cid + '\t' + ','.join(frags_id) + '\n')
                with open(result_fp2, 'a') as f_handle2:
                    f_handle2.write(' '.join(frags_id) + '\n')
            else:
                frags_id = frags
                with open(result_fp2, 'a') as f_handle2:
                    f_handle2.write(' '.join(frags_id) + '\n')
            counter += 1


def count_fragment(cid2frag_fp):
    """
    count fragment in all training set
    examples: 10	CN,C1=CNC=NC1,C1=CNCCN1,CC,CN,N,CN,C1=CC=CC=C1,CC,C,CN,CN,C,CC,CC,CC,C,C=O
    :param cid2frag_fp: cid2fragment file path, i.e. step2_result file
    :return:
    """
    frag2num = {}
    with open(cid2frag_fp, 'r') as f_handle:
        counter = 0
        for i in f_handle:
            if counter % 500000 == 0:
                t = datetime.datetime.now()
                print('>>> Current line: {}'.format(counter), t.strftime("%c"))
            cid, sentence = i.strip().split('\t')
            frags = sentence.split(',')
            for frag in frags:
                if frag not in frag2num:
                    frag2num[frag] = 0
                frag2num[frag] += 1
            counter += 1
    frag2num_df = pd.DataFrame.from_dict(frag2num, orient='index')
    frag2num_df.sort_values(by=0, inplace=True, ascending=False)
    frag2num_df.reset_index(inplace=True)
    frag2num_df.rename(columns={0: 'count', 'index': 'fragment'}, inplace=True)
    return frag2num_df


def get_smiles(mol):
    """
    mol obj -> SMILES
    :param mol:
    :return:
    """
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def get_mol(smiles):
    """
    SMILES -> mol obj
    :param smiles:
    :return:
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol


class Refragment(object):
    def __init__(self, g, f2f, smiles, test=False):
        """
        g: moelcular graph created by networkx with SMILES and mol_inx for each node
        f2f: fragment2frequency, a dataframe with fragment(SMILES) as index, count/frequency, same as f2n
        smiles: the SMILES of the whold molecule
        mol_inx means the index of each atom in the whole molecule, it's a unique id for each atom
        """
        self.g = g
        self.smiles = smiles
        self.f2f = f2f
        self.test = test

    def get_node_by_degree(self, d=1):
        """
        # The node degree is the number of edges adjacent to the node
        degree equals to 1 means all leaves on the end of graph
        :param d: degree, 1/2
        :return: return all nodes with specific degree
        """
        node_degree = dict(self.g.degree)
        return [k for k, v in node_degree.items() if v == d]

    def get_degree_by_node(self, node_id):
        node_degree = dict(self.g.degree)
        return node_degree[node_id]

    def get_neighbors(self, node_id):
        neigs = list(self.g.neighbors(node_id))
        return neigs

    def check_if_merge(self, node1_id, node2_id):
        """
        check if need to merge these two nodes depend on the frequency of each node
        """
        mean_freq = self.get_mean_frequency()
        node1_freq = self.get_freq(node1_id)
        node2_freq = self.get_freq(node2_id)
        if self.test:
            print('  >type of each element:', type(mean_freq), type(node1_freq), type(node2_freq))
        if (node1_freq >= mean_freq) and (node2_freq >= mean_freq):
            return True
        return False

    def get_freq(self, node_id):
        """
        get frequency by node id
        """
        smiles = self.get_node_attr(node_id, 'smiles')
        # print('current fragment SMILES is: {}'.format(smiles))
        # print(self.f2f.loc[smiles])
        return self.f2f.loc[smiles, 'frequency']

    def get_node_attr(self, node_id, attr):
        """
        get node attribute by node id
        attr: smiles/mol_inx
        """
        if self.test:
            print(node_id)
            print(type(self.g.nodes[node_id]))
        return self.g.nodes[node_id].get(attr, '')

    # def set_node_attr()

    def get_mean_frequency(self, min_count=3):
        """
        mean of the frequency for all fragments which count >= min_count
        """
        mean_freq = self.f2f.loc[self.f2f['count'] >= min_count, 'frequency'].mean()
        return mean_freq

    def _merge_smiles(self, node1_id, node2_id):
        node1_inx_cluster = self.get_node_attr(node1_id, 'mol_inx')
        node2_inx_cluster = self.get_node_attr(node2_id, 'mol_inx')
        if self.test:
            print('  >The mol_inx of node {} is {}'.format(node1_id, node1_inx_cluster))
            print('  >The mol_inx of node {} is {}'.format(node2_id, node2_inx_cluster))
        inx_cluster = set(node1_inx_cluster) | set(node2_inx_cluster)
        merged_smiles = self._get_smiles_by_inx(inx_cluster)
        return {'merged_smiles': merged_smiles, 'merged_inx': inx_cluster}

    def merge_two_nodes(self, left_id, right_id):
        """
        remove right node to left node, and right_id will be delete;
        merge SMILES of these two nodes;
        add new fragment to self.f2f;
        update count and frequency in self.f2f
        """
        if self.check_if_merge(left_id, right_id):
            raw_smiles_left = self.g.nodes[left_id]['smiles']
            raw_smiles_right = self.g.nodes[right_id]['smiles']
            g2 = nx.contracted_nodes(self.g, left_id, right_id, self_loops=False)
            merged_result = self._merge_smiles(left_id, right_id)
            merged_smiles = merged_result['merged_smiles']
            g2.nodes[left_id]['smiles'] = merged_smiles
            g2.nodes[left_id]['mol_inx'] = list(merged_result['merged_inx'])
            if self.test:
                print('  >Merged result is: {}'.format(merged_result))
                print('  >New network: {}'.format(g2.nodes(data=True)))

            if merged_smiles not in self.f2f.index:
                self.f2f.loc[merged_smiles, 'count'] = 0
            self.f2f.loc[merged_smiles, 'count'] += 1
            self.f2f.loc[raw_smiles_left, 'count'] -= 1
            self.f2f.loc[raw_smiles_right, 'count'] -= 1
            self.f2f['frequency'] = self.f2f['count'] / self.f2f['count'].sum()
            self.g = g2.copy()

    def _get_mol(self):
        """
        SMILES -> mol obj
        :return:
        """
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            return None
        Chem.Kekulize(mol)
        return mol

    def _get_smiles_by_inx(self, inx_cluster):
        """
        get a subset smiles in the whole molecule by inx_cluster
        :param inx_cluster: a set of atom index in molecule, at least contains two elements
        :return:
        """
        mol = self._get_mol()
        if self.test:
            print('  >atom index cluster: {}'.format(inx_cluster))
        smiles = Chem.MolFragmentToSmiles(mol, inx_cluster, kekuleSmiles=True)
        new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
        # new_mol = copy_edit_mol(new_mol).GetMol()
        new_mol = sanitize(new_mol)  # We assume this is not None
        return get_smiles(new_mol)

    def update(self):
        """
        main part of this class
        find all leaves (only have one neighbor) and merge with their neighbor if needed
        """
        for d in range(1, 3):
            # d is 1 or 2
            if self.test:
                print('---------------------------------degree {}--------------------------'.format(d))
            nodes = self.get_node_by_degree(d=d)  # a list of node id
            for node in nodes:
                if node in list(self.g.nodes):
                    neighbors = self.get_neighbors(node)  # a list of node id
                    if self.test:
                        print()
                        print('## Current node is: {}'.format(node))
                        print('  >>> Neighbors of this node are : {}'.format(','.join([str(i) for i in neighbors])))
                    for neighbor in neighbors:
                        # neighbor may be deleted on this process, so need to check if it exists
                        if d == 1:  # degree = 1, only leaves
                            if self.test:
                                print('  >>> Start to check if {} and {} can be merged...'.format(neighbor, node))
                            if (neighbor in list(self.g.nodes)) and self.check_if_merge(neighbor, node):
                                if self.test:
                                    print('  >>> Start to merge {} to {}...'.format(node, neighbor))
                                self.merge_two_nodes(left_id=neighbor, right_id=node)
                        if d == 2:  # degree = 2, only merge with the neighbor which degree is 2
                            if self.get_degree_by_node(neighbor) == 2:
                                if self.test:
                                    print('    >the degree of neighbor {} is {}'.format(
                                        neighbor, self.get_degree_by_node(neighbor)))
                                    print('  >>> Start to check if {} and {} can be merged...'.format(neighbor, node))
                                if (neighbor in list(self.g.nodes)) and self.check_if_merge(neighbor, node):
                                    if self.test:
                                        print('  >>> Start to merge {} to {}...'.format(neighbor, node))
                                    self.merge_two_nodes(left_id=node, right_id=neighbor)

        n2n = {n: list(self.g.neighbors(n)) for n in list(self.g.nodes())}  # node 2 neighbors, {id: [], ... }
        id2smiles = nx.get_node_attributes(self.g, 'smiles')
        id2mol_inx = nx.get_node_attributes(self.g, 'mol_inx')
        return {'n2n': n2n, 'id2smiles': id2smiles, 'f2f': self.f2f, 'id2mol_inx': id2mol_inx}

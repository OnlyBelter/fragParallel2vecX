import os
from mordred import descriptors
from .._base import (get_mol, cal_md_by_smiles, MAIN_ELEMENT)

# MAIN_ELEMENT = ['C', 'O', 'N', 'H', 'P', 'S', 'Cl', 'F', 'Br', 'I']


def check_only_main_element(smiles):
    only_main_element = True
    fr_mol = get_mol(smiles)
    if fr_mol:
        for atom in fr_mol.GetAtoms():
            if atom.GetSymbol() not in MAIN_ELEMENT:
                only_main_element = False
                break
    else:
        return False
    return only_main_element


def get_atom_formal_charge(smiles):
    mol = get_mol(smiles)
    if mol:
        # CID: 10553846 has two positive charges and two negative charges
        return sum([abs(atom.GetFormalCharge()) for atom in mol.GetAtoms()])
    else:
        return -1


def get_md(smiles):
    """

    :param smiles:
    :return: ['MW', 'SLogP', 'nRot'] values, a dataframe
    """
    md_list = ['MW', 'SLogP', 'nRot']
    desc = [descriptors.Weight, descriptors.RotatableBond, descriptors.SLogP]
    md_values = cal_md_by_smiles(smiles, md_list, desc=desc)
    return md_values


def do_step2(input_fp, output_fp):
    """
    get MD and filter by charge and element
    :param input_fp:
    :param output_fp:
    :return:
    """
    output_cols = ['CID', 'MW', 'SLogP', 'nRot']
    # data_dir = '../demo_data'
    # input_fp = input_fp
    # output_fp = output_fp
    if not os.path.exists(output_fp):
        with open(output_fp, 'w') as output_fh:
            output_fh.write(','.join(output_cols) + '\n')
    # https://stackoverflow.com/questions/42339876/error-unicodedecodeerror-utf-8-codec-cant-decode-byte-0xff-in-position-0-in
    with open(input_fp, 'r', encoding='utf8') as input_fh:
        counter = 1
        smiles2cid = {}
        for each_line in input_fh:
            # print(each_line)
            # each_line = each_line
            if counter >= 1:
                each_line = each_line.strip()
                cid, _smiles = each_line.split('\t')
                sum_charge = get_atom_formal_charge(_smiles)
                if_only_main_ele = check_only_main_element(_smiles)
                if sum_charge == 0 and if_only_main_ele:
                    smiles2cid[_smiles] = cid
                    if counter % 10000 == 0:
                        print('>>> current line: {}'.format(counter))
                        _mol_info = get_md(list(smiles2cid.keys()))
                        # print('_mol_info', _mol_info)
                        _mol_info['CID'] = list(smiles2cid.values())
                        _mol_info = _mol_info[['CID', 'MW', 'SLogP', 'nRot']]
                        _mol_info.to_csv(output_fp, mode='a', header=None, index=False, float_format='%.3f')
                        smiles2cid = {}
                else:
                    # print(cid, sum_charge, if_only_main_ele)
                    pass
            counter += 1
        if len(smiles2cid) != 0:
            _mol_info = get_md(list(smiles2cid.keys()))
            # print('_mol_info', _mol_info)
            _mol_info['CID'] = list(smiles2cid.values())
            _mol_info = _mol_info[['CID', 'MW', 'SLogP', 'nRot']]
            _mol_info.to_csv(output_fp, mode='a', header=None, index=False, float_format='%.3f')


def filter_by_md(input_fp, output_fp, md2threshold=None):
    """
    filter molecules by MD (200<= MW <= 600, -10<= SLogP <= 10, nRot <= 30)
    :param input_fp: file path
    :param output_fp:
    :param md2threshold: {'MW': (200, 600), 'SLogP': (-10, 10), 'nRot': (0, 30)}
    :return:
    """
    if not md2threshold:
        md2threshold = {'MW': (200, 600), 'SLogP': (-10, 15), 'nRot': (0, 40)}
    with open(input_fp, 'r') as input_fh:
        header = input_fh.readline()
        for each_line in input_fh:
            each_line = each_line.strip()
            cid, mw, logp, nrot = each_line.split(',')
            keep_line = False
            if (md2threshold['MW'][0] <= float(mw) <= md2threshold['MW'][1]) & \
                    (md2threshold['SLogP'][0] <= float(logp) <= md2threshold['SLogP'][1]) & \
                    (md2threshold['nRot'][0] <= int(nrot) <= md2threshold['nRot'][1]):
                keep_line = True
            if keep_line:
                with open(output_fp, 'a') as output_f_handle:
                    output_f_handle.write(each_line + '\n')


if __name__ == '__main__':
    data_dir = '../../big_data'
    input_fp = os.path.join(data_dir, '01_raw_data', 'cid2SMILES.txt')
    output_fp = os.path.join(data_dir, '02_filtered_molecule', 'cid2md_filtered_by_ele_and_charge.txt')
    output_fp_filtered_by_md = os.path.join(data_dir, '02_filtered_molecule', 'cid2md_filtered_by_md.txt')
    do_step2(input_fp=input_fp, output_fp=output_fp)

    filter_by_md(input_fp=output_fp, output_fp=output_fp_filtered_by_md)

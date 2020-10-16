# PubChem PUG REST
# http://pubchemdocs.ncbi.nlm.nih.gov/pug-rest
# https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest
import io
import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from ..utility import PUBCHEM_BASE_URL

# PUBCHEM_BASE_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{26856757}/property/{}/CSV'
# BASE_URL2 = 'http://classyfire.wishartlab.com/entities/{}.json'


def get_property_by_cid(cids, mol_property, result_fn, log_fn):
    """

    :param cids: a single cid or multiple cids sperated by ","
    :param mol_property: IsomericSMILES/ CanonicalSMILES/ MolecularFormula/ MolecularWeight/ InChIKey
    :param result_fn: filename of result
    :param log_fn: filename of log
    :return:
    """
    first_cid = cids.split(',')[0]
    try:
        result = requests.get(PUBCHEM_BASE_URL.format(cids, mol_property))
        if result.status_code == 200:
            # https://stackoverflow.com/a/32400969/2803344
            content = result.content
            c = pd.read_csv(io.StringIO(content.decode('utf-8')))
            # print(c.to_dict())
            for i in range(c.shape[0]):
                with open(result_fn, 'a') as f:
                    cid = c.loc[i, 'CID']
                    prop_value = c.loc[i, mol_property]
                    try:
                        f.write('\t'.join([str(cid), prop_value]) + '\n')
                    except:
                        print(cid, prop_value)
                        with open(log_fn, 'a') as log_f:
                            # '3' means that this cid doesn't exist (no SMILES)
                            log_f.write('3' + '\t' + str(cid) + '\n')
            with open(log_fn, 'a') as log_f:
                log_f.write('1' + '\t' + first_cid + '\n')  # '1' means success
    except:
        print('Getting the {} of cid {} is not successful.'.format(property, cids))
        with open(log_fn, 'a') as log_f:
            log_f.write('0' + '\t' + first_cid + '\n')  # '0' means failed


def call_download_cid2smiles(max_cid=100, interval=200,
                             result_file_dir=None, cid_list=None,
                             file_name=None):
    """

    :param max_cid: int
        CIDs from 1 to max_cid will be downloaded
    :param interval: int
        how many CIDs need to download each query
    :param result_file_dir: str
    :param cid_list: list
        download smiles by CID list directly
    :param file_name: str

    :return:
    """
    if not os.path.exists(result_file_dir):
        os.makedirs(result_file_dir)
    result_file_path = os.path.join(result_file_dir, 'cid2SMILES.txt')
    if file_name is not None:
        result_file_path = os.path.join(result_file_dir, file_name)
    log_file_path = os.path.join(result_file_dir, 'cid2SMILES_download_status_class.log')
    # max_cid = 30000001
    # interval = 200

    if cid_list is not None:
        max_cid = len(cid_list)
    if interval < max_cid:
        interval = max_cid
    for cid in range(1, max_cid + 1, interval):    # cid or cid_inx
        success_first_cids = {}
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as log_f:
                for each_line in log_f:
                    each_line = each_line.strip().split('\t')
                    each_line = [int(i) for i in each_line]
                    if each_line[0] == 1:
                        success_first_cids[each_line[1]] = 1
        # print(len(success_first_cids))
        # print(cid, list(success_first_cids.keys())[:10])
        if cid not in success_first_cids.keys():
            if cid_list is not None:
                current_cid_list = cid_list[cid-1:cid+interval]
                print('Downloading {} molecules, CID: {} ... {}'.format(len(current_cid_list),
                                                                        current_cid_list[0], current_cid_list[-1]))
            else:
                current_cid_list = list(range(cid, cid + interval))
                print('Downloading cid from {} to {}'.format(current_cid_list[0], current_cid_list[-1]))
            print(current_cid_list)
            _cids = ','.join([str(i) for i in current_cid_list])
            # print(_cids)
            time.sleep(3)
            get_property_by_cid(_cids, 'CanonicalSMILES', result_fn=result_file_path, log_fn=log_file_path)


def remove_duplicate_smiles(cid2smiles_file_path, output_file_path):
    """

    :param cid2smiles_file_path:
    :param output_file_path:
    :return:
    """
    smiles2cid = {}
    if not os.path.exists(output_file_path):
        with open(output_file_path, 'w') as f:
            f.write('\t'.join(['cid', 'smiles']) + '\n')
    counter = 0
    with open(cid2smiles_file_path, 'r') as f:
        for line in tqdm(f):
            counter += 1
            cid, smiles = line.strip().split('\t')
            if (cid != 'cid') and ('.' not in smiles):
                smiles2cid[smiles] = cid
    print('>>> The number of total lines before filtered: {}'.format(counter))
    print('>>> The number of unique SMILES: {}'.format(len(smiles2cid)))
    with open(output_file_path, 'a') as f:
        for smiles, cid in tqdm(smiles2cid.items()):
            f.write('\t'.join([cid, smiles]) + '\n')


if __name__ == '__main__':
    cid_file = 'big-data/mol2class_pca_kmeans_1000.csv'

# PubChem PUG REST
# http://pubchemdocs.ncbi.nlm.nih.gov/pug-rest
# https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest
import io
import time
import requests
import pandas as pd
from .._base import PUBCHEM_BASE_URL

# PUBCHEM_BASE_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/{}/CSV'
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


if __name__ == '__main__':
    cid_file = 'big-data/mol2class_pca_kmeans_1000.csv'
    result_file = '../../big_data/01_raw_data/cid2SMILES.txt'
    log_file = '../../big_data/01_raw_data/cid2SMILES_download_status_class.log'
    max_cid = 30000001
    interval = 200

    for cid in range(1, max_cid, interval):
        success_first_cids = {}
        with open(log_file, 'r') as log_f:
            for each_line in log_f:
                each_line = each_line.strip().split('\t')
                each_line = [int(i) for i in each_line]
                if each_line[0] == 1:
                    success_first_cids[each_line[1]] = 1
        # print(len(success_first_cids))
        # print(cid, list(success_first_cids.keys())[:10])
        if cid not in success_first_cids.keys():
            cid_list = list(range(cid, cid+interval))
            print('Downloading cid from {} to {}'.format(cid_list[0], cid_list[-1]))
            print(cid_list)
            _cids = ','.join([str(i) for i in cid_list])
                # print(_cids)
            time.sleep(3)
            get_property_by_cid(_cids, 'CanonicalSMILES', result_fn=result_file, log_fn=log_file)

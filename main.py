import os
from fragt2vec._prepare_data import do_step2


# step1 get smiles by cid
data_dir = 'fragt2vec/demo_data'

# step2 filter molecules by MD
input_fp = os.path.join(data_dir, '01_raw_data', 'cid2SMILES_head100.txt')
output_fp = os.path.join(data_dir, '02_filtered_molecule', 'cid2md_before_filter.txt')
do_step2(input_fp=input_fp, output_fp=output_fp)


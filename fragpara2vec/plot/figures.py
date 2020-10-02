import os

import argparse
import pandas as pd

from fragpara2vec.utility import SELECTED_MD, get_format_time, find_nearest_neighbor, draw_multiple_mol, show_each_md, reduce_by_tsne

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

import rdkit
from rdkit.Chem.Draw import IPythonConsole
# IPythonConsole.ipython_useSVG = True
from IPython.display import SVG
from sklearn.decomposition import PCA

import rdkit.Chem as Chem
from rdkit.Chem import Draw



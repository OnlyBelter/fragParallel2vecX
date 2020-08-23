# fragTandem2vecX
New repository of fragTandem2vec, fragmentation and vectorization of small molecules

## 1. Getting started (tested on Ubuntu 18.04 and Windows 10)
### 1.1 download Miniconda and create new env
- download miniconda, please see https://conda.io/en/master/miniconda.html
- also see: [Building identical conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments)
```shell script
conda create --name fragTandem2vec python=3.6
conda activate fragTandem2vec
```

### 1.2 install fragt2vec
```shell script
git clone https://github.com/OnlyBelter/fragTandem2vecX.git
cd fragTandem2vecX
pip install .
```

### 1.3 install rdkit by conda
```shell script
# install rdkit and mol2vec
conda install -c rdkit rdkit==2019.03.3.0
```

## 2. Using fragt2vec
# TODO

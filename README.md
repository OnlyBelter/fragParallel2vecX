# fragTandem2vecX
New repository of fragTandem2vec, fragmentation and vectorization of small molecules

## 1. Getting started (tested on Ubuntu 18.04 )
### 1.1 clone github repo
```shell script
git clone https://github.com/OnlyBelter/fragTandem2vecX.git
```

### 1.2 download Miniconda and install dependencies
- download miniconda, please see https://conda.io/en/master/miniconda.html
- also see: [Building identical conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments)
```shell script
cd /where/is/fragTandem2vecX/located
conda create --name fragTandem2vec --file requirements.txt
```

### 1.3 activate environment just created
```shell script
conda activate fragTandem2vec

# install rdkit and mol2vec
conda install -c rdkit rdkit==2019.03.3.0
# TODO, merge key scripts into my repo to replace mol2vec
pip install git+https://github.com/samoturk/mol2vec

# install mordred
pip install git+https://github.com/mordred-descriptor/mordred

# install tensorflow
pip install tensorflow==2.1.0
```

### 1.4 building fastText for Python
# TODO: replace fasttext by gensim
- also see: https://github.com/facebookresearch/fastText#building-fasttext-for-python
```shell script
$ pip install git+https://github.com/facebookresearch/fastText
```

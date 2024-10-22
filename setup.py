import io
import os
import sys

from setuptools import setup, find_packages

install_requires = ["networkx==2.4", "matplotlib==3.1.0", "pandas", "scikit-learn",
                    "seaborn", "requests", "tensorflow==2.1", "scipy==1.4.1", "numpy",
                    'gensim==3.8.3', "mordred==1.2.0", "tqdm", "joblib"]

if sys.version_info < (3, 4, 0):
    install_requires.append("enum34")


def get_version():
    with open(os.path.join("fragpara2vec", "_version.txt")) as f:
        return f.read().strip()


def get_test_data():
    for p, _, fs in os.walk(os.path.join("fragpara2vec", "tests", "references")):
        p = p.split(os.sep)[2:]

        for f in fs:
            yield os.path.join(*(p + [f]))


README_rst = ""
fndoc = os.path.join(os.path.dirname(__file__), "README.rst")
with io.open(fndoc, mode="r", encoding="utf-8") as fd:
    README_rst = fd.read()

setup(
    name="fragpara2vec",
    version=get_version(),
    description="molecular vector calculator",
    long_description=README_rst,
    license="BSD-3-Clause",
    author="Xin (Belter) Xiong",
    author_email="onlybelter@outlook.com",
    url="",
    platforms=["any"],
    keywords="QSAR chemoinformatics",
    packages=find_packages(),
    package_data={
        "fragpara2vec": ["demo_data/*.txt", "_version.txt"],
        "fragpara2vec.tests": list(get_test_data()),
    },
    install_requires=install_requires,
    tests_require=["nose==1.*", "PyYaml>=4.2b1"],
    extras_require={"full": ["pandas", "tqdm"]},
    cmdclass={"test": None},
)

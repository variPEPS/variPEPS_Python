
# variPEPS -- Versatile tensor network library for variational ground state simulations in two spatial dimensions.

[![DOI](https://zenodo.org/badge/773767511.svg)](https://zenodo.org/doi/10.5281/zenodo.10852390)
[![Documentation Status](https://readthedocs.org/projects/varipeps/badge/?version=latest)](https://varipeps.readthedocs.io/en/stable/?badge=latest)

variPEPS is the Python variant of the tensor network library developed for
variational ground state simulations in two spatial dimensions applying gradient
optimization using automatic differentation.

For a detailed report on the method, please see our publication currently available as preprint on arXiv: [https://arxiv.org/abs/2308.12358](https://arxiv.org/abs/2308.12358).

## Installation
### Installation using pip
The current version of the variPEPS Python package is available on [PyPI](https://pypi.org/project/variPEPS/). It can be easily installed by using the Python package manager pip:
```bash
$ python3 -m pip install variPEPS
```

### Installation using poetry

The dependencies in this project are managed by poetry and the tool can also be used to install the package including a fixed set of dependencies with a specific version. For more details how poetry is operating, please see the [upstream documentation](http://python-poetry.org/docs/).

To install dependencies you can just run in the main folder of the variPEPS project:
```bash
$ poetry install
```
or if you do not need the development packages:
```bash
$ poetry install --no-dev
```

## Usage

For detailed information how to use the package we want to point out to the [documentation of the project](https://varipeps.readthedocs.io/en/stable).

## Citation

We are happy if you want to use the framework for your research. For the citation of our work we ask to use the following references (the publication with the method description, the Zenodo reference for this Git repository and the repository itself):
* J. Naumann, E. L. Weerda, M. Rizzi, J. Eisert, and P. Schmoll, variPEPS -- a versatile tensor network library for variational ground state simulations in two spatial dimensions (2023), [arXiv:2308.12358](https://arxiv.org/abs/2308.12358).
* J. Naumann, P. Schmoll, F. Wilde, and F. Krein, [variPEPS (Python version)](https://zenodo.org/doi/10.5281/zenodo.10852390), Zenodo.

The BibTeX code for these references are:
```bibtex
@misc{naumann23_varipeps,
    title =         {variPEPS -- a versatile tensor network library for variational ground state simulations in two spatial dimensions},
    author =        {Jan Naumann and Erik Lennart Weerda and Matteo Rizzi and Jens Eisert and Philipp Schmoll},
    year =          {2023},
    eprint =        {2308.12358},
    archivePrefix = {arXiv},
    primaryClass =  {cond-mat.str-el}
}

@software{naumann24_varipeps_python,
    author =        {Jan Naumann and Philipp Schmoll and Frederik Wilde and Finn Krein},
    title =         {{variPEPS (Python version)}},
    howpublished =  {Zenodo},
    url =           {https://doi.org/10.5281/ZENODO.10852390},
    doi =           {10.5281/ZENODO.10852390},
}
```

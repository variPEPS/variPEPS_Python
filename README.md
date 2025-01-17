
# variPEPS -- Versatile tensor network library for variational ground state simulations in two spatial dimensions.

[![DOI](https://zenodo.org/badge/773767511.svg)](https://zenodo.org/doi/10.5281/zenodo.10852390)
[![Documentation Status](https://readthedocs.org/projects/varipeps/badge/?version=latest)](https://varipeps.readthedocs.io/en/stable/?badge=latest)
[![PyPI - Version](https://img.shields.io/pypi/v/varipeps)](https://pypi.org/project/variPEPS/)

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

## Usage

For detailed information how to use the package we want to point out to the [documentation of the project](https://varipeps.readthedocs.io/en/stable).

## Citation

We are happy if you want to use the framework for your research. For the citation of our work we ask to use the following references (the publication with the method description, the Zenodo reference for this Git repository and the repository itself):
* J. Naumann, E. L. Weerda, M. Rizzi, J. Eisert, and P. Schmoll, An introduction to infinite projected entangled-pair state methods for variational ground state simulations using automatic differentiation, SciPost Phys. Lect. Notes 86 (2024), doi:[10.21468/SciPostPhysLectNotes.86](https://doi.org/10.21468/SciPostPhysLectNotes.86).
* J. Naumann, P. Schmoll, F. Wilde, and F. Krein, [variPEPS (Python version)](https://zenodo.org/doi/10.5281/zenodo.10852390), Zenodo.

The BibTeX code for these references are:
```bibtex
@article{10.21468/SciPostPhysLectNotes.86,
	title={{An introduction to infinite projected entangled-pair state methods for variational ground state simulations using automatic differentiation}},
	author={Jan Naumann and Erik Lennart Weerda and Matteo Rizzi and Jens Eisert and Philipp Schmoll},
	journal={SciPost Phys. Lect. Notes},
	pages={86},
	year={2024},
	publisher={SciPost},
	doi={10.21468/SciPostPhysLectNotes.86},
	url={https://scipost.org/10.21468/SciPostPhysLectNotes.86},
}

@software{naumann24_varipeps_python,
    author =        {Jan Naumann and Philipp Schmoll and Frederik Wilde and Finn Krein},
    title =         {{variPEPS (Python version)}},
    howpublished =  {Zenodo},
    url =           {https://doi.org/10.5281/ZENODO.10852390},
    doi =           {10.5281/ZENODO.10852390},
}
```

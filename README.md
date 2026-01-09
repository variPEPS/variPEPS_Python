![Logo of the variPEPS library with a triangular-PEPS norm as picture and "variPEPS -- Variational PEPS Library" as text right of it](https://github.com/variPEPS/variPEPS_Python/raw/main/docs/source/images/logo.png)

# variPEPS -- Versatile tensor network library for variational ground state simulations in two spatial dimensions.

[![DOI](https://zenodo.org/badge/773767511.svg)](https://zenodo.org/doi/10.5281/zenodo.10852390)
[![Documentation Status](https://readthedocs.org/projects/varipeps/badge/?version=latest)](https://varipeps.readthedocs.io/en/stable/?badge=latest)
[![PyPI - Version](https://img.shields.io/pypi/v/varipeps)](https://pypi.org/project/variPEPS/)

variPEPS is the Python variant of the tensor network library developed for
variational ground state simulations in two spatial dimensions applying gradient
optimization using automatic differentation.

For a detailed report on the method, please see our publications listed at the
end of this readme.

## Installation
### Installation using pip
The current version of the variPEPS Python package is available on
[PyPI](https://pypi.org/project/variPEPS/).
It can be easily installed by using the Python package manager pip:
```bash
$ python3 -m pip install variPEPS
```

## Usage

For detailed information how to use the package we want to point out to the
[documentation of the project](https://varipeps.readthedocs.io/en/stable).

## Citation

We are happy if you want to use the framework for your research. For the
citation of our work we ask to use the following references (the publications
with the method description and the Zenodo reference for this Git repository):
* J. Naumann, E. L. Weerda, M. Rizzi, J. Eisert, and P. Schmoll, An introduction
  to infinite projected entangled-pair state methods for variational ground
  state simulations using automatic differentiation, SciPost Phys. Lect. Notes
  86 (2024),
  doi:[10.21468/SciPostPhysLectNotes.86](https://doi.org/10.21468/SciPostPhysLectNotes.86).
* J. Naumann, E. L. Weerda, J. Eisert, M. Rizzi and P. Schmoll, Variationally
  optimizing infinite projected entangled-pair states at large bond dimensions:
  A split corner transfer matrix renormalization group approach, Phys. Rev. B
  111, 235116 (2025),
  doi:[10.1103/PhysRevB.111.235116](https://doi.org/10.1103/PhysRevB.111.235116).
* J. Naumann, J. Eisert, P. Schmoll, Variational optimization of projected
  entangled-pair states on the triangular lattice,
  [arXiv:2510.04907](https://arxiv.org/abs/2510.04907)
* J. Naumann, P. Schmoll, R. Losada, F. Wilde, and F. Krein, [variPEPS (Python
  version)](https://zenodo.org/doi/10.5281/zenodo.10852390), Zenodo.

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

@article{PhysRevB.111.235116,
    title = {Variationally optimizing infinite projected entangled-pair states at large bond dimensions: A split corner transfer matrix renormalization group approach},
    author = {Naumann, Jan and Weerda, Erik L. and Eisert, Jens and Rizzi, Matteo and Schmoll, Philipp},
    journal = {Phys. Rev. B},
    volume = {111},
    issue = {23},
    pages = {235116},
    numpages = {12},
    year = {2025},
    month = {Jun},
    publisher = {American Physical Society},
    doi = {10.1103/PhysRevB.111.235116},
    url = {https://link.aps.org/doi/10.1103/PhysRevB.111.235116}
}

@misc{naumann2025variationaloptimizationprojectedentangledpair,
    title={Variational optimization of projected entangled-pair states on the triangular lattice},
    author={Jan Naumann and Jens Eisert and Philipp Schmoll},
    year={2025},
    eprint={2510.04907},
    archivePrefix={arXiv},
    primaryClass={cond-mat.str-el},
    url={https://arxiv.org/abs/2510.04907},
}

@software{naumann_varipeps_python,
    author =        {Jan Naumann and Philipp Schmoll and Roberto Losada and Frederik Wilde and Finn Krein},
    title =         {{variPEPS (Python version)}},
    howpublished =  {Zenodo},
    url =           {https://doi.org/10.5281/ZENODO.10852390},
    doi =           {10.5281/ZENODO.10852390},
}
```

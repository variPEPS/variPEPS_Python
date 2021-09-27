# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'peps-ad'
copyright = '2021, Philipp Schmoll, Augustine Kshetrimayum, Frederik Wilde, Jan Naumann'
author = 'Philipp Schmoll, Augustine Kshetrimayum, Frederik Wilde, Jan Naumann'

# The full version, including alpha/beta/rc tags
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
]

napoleon_include_private_with_doc = True
napoleon_preprocess_types = False

autodoc_type_aliases = {
    "T_PEPS_Unit_Cell": "PEPS_Unit_Cell",
    "T_PEPS_Tensor": "PEPS_Tensor",
    "PEPS_Unit_Cell.Unit_Cell_Data": "peps_ad.peps.PEPS_Unit_Cell.Unit_Cell_Data",
}

autodoc_mock_imports = ["jax"]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

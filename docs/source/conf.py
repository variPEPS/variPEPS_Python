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
import importlib
import importlib.metadata
import inspect

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "varipeps"
copyright = "2021-2024, Jan Naumann, Philipp Schmoll, Frederik Wilde, Finn Krein"
author = "Jan Naumann, Philipp Schmoll, Frederik Wilde, Finn Krein"

# The full version, including alpha/beta/rc tags
release = importlib.metadata.version("varipeps")


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_defaultargs",
    "sphinx_subfigure",
]

napoleon_include_private_with_doc = True
napoleon_preprocess_types = False

autodoc_type_aliases = {
    "T_PEPS_Unit_Cell": "PEPS_Unit_Cell",
    "T_PEPS_Tensor": "PEPS_Tensor",
    "PEPS_Unit_Cell.Unit_Cell_Data": "varipeps.peps.PEPS_Unit_Cell.Unit_Cell_Data",
    "jax._src.numpy.lax_numpy.ndarray": "jax.numpy.ndarray",
    "h5py._hl.group.Group": "h5py.Group",
}

# autodoc_mock_imports = ["jax"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

from varipeps import git_commit, git_tag

code_url = "https://github.com/variPEPS/variPEPS_Python/blob/{blob}"

if git_tag is not None:
    code_url = code_url.format(blob=git_tag)
elif git_commit is not None:
    code_url = code_url.format(blob=git_commit)
else:
    code_url = code_url.format(blob=release)


def linkcode_resolve(domain, info):
    # Code adapted from function in websockets module
    # https://github.com/python-websockets/websockets/blob/e217458ef8b692e45ca6f66c5aeb7fad0aee97ee/docs/conf.py

    if domain != "py":
        return None
    if not info["module"]:
        return None

    mod = importlib.import_module(info["module"])
    if "." in info["fullname"]:
        objname_parts = info["fullname"].split(".")

        obj = getattr(mod, objname_parts[0])

        for name in objname_parts[1:-1]:
            obj = getattr(obj, name)

        attrname = objname_parts[-1]

        try:
            # object is a method of a class
            obj = getattr(obj, attrname)
        except AttributeError:
            # object is an attribute of a class
            return None
    else:
        obj = getattr(mod, info["fullname"])

    try:
        filename = inspect.getsourcefile(obj)
        lines = inspect.getsourcelines(obj)
    except (TypeError, OSError):
        # e.g. object is a typing.Union
        return None

    filename = os.path.relpath(filename, os.path.abspath("../.."))

    if not filename.startswith("varipeps"):
        # e.g. object is a typing.NewType
        return None
    start, end = lines[1], lines[1] + len(lines[0]) - 1

    return f"{code_url}/{filename}#L{start}-L{end}"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

rst_prolog = """
.. |default| raw:: html

    <div class="default-value-section"> <span class="default-value-label">Default:</span>"""

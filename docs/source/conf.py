# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../'))  # Adjust the path accordingly

# -- Project information -----------------------------------------------------
project = 'ttta: Tools for temporal text analysis in Python '
copyright = '2024, Kai-Robin Lange'
author = 'Kai-Robin Lange, Lars Grönberg, Niklas Benner, Imene Kolli, Aymane Hachcham, Jonas Rieger and Carsten Jenstsch'
release = '0.9.0'

# -- General configuration ---------------------------------------------------
extensions = ['sphinx.ext.autodoc']  # Added the autodoc extension

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'
html_static_path = ['_static']

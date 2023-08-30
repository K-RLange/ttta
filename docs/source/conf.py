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
project = 'tta'
copyright = '2023, TU Dortmund'
author = 'TU Dortmund'
release = '1'

# -- General configuration ---------------------------------------------------
extensions = ['sphinx.ext.autodoc']  # Added the autodoc extension

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'
html_static_path = ['_static']

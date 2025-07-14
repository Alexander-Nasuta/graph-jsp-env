# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'graph-jsp-env'
copyright = '2025, Alexander Nasuta'
author = 'Alexander Nasuta'
release = '1.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.duration",
    "sphinx_copybutton",

    "sphinx.ext.autosectionlabel",

    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",

    "sphinx.ext.autodoc",

    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "sphinx.ext.viewcode",

    "nbsphinx",

    'sphinxcontrib.bibtex',
    'sphinxcontrib.tikz',
    'sphinx.ext.mathjax'

]

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    # other extensions
]

bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'alpha'

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_title = f"Graph Job Shop Environment"
html_static_path = ['_static']
html_theme_options = {
    "light_logo": "logo-graph-light.png",
    "dark_logo": "logo-graph-dark.png",
}
html_css_files = [
    'custom.css',
]

# Configure autosectionlabel
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

# Exclude specific files
autosectionlabel_exclude_files = ['README.md']
exclude_patterns = [
    'README.md'
]

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'preamble': r'''
\usepackage[utf8]{inputenc}
''',
    'figure_align': 'htbp',
}


# Sphinx configuration for PEM Electrolyzer PINN MLOPS
project = 'Physics-Informed Neural Networks for PEM Electrolyzers'
copyright = '2026, NAIC / NORCE / Sigma2'
author = 'NAIC Team'
release = '1.0'

html_logo = 'images/NRIS-Logo.png'

extensions = ['sphinxcontrib.mermaid', 'sphinx_lesson', 'sphinx.ext.githubpages', 'sphinx_tabs.tabs']
templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
sphinx_tabs_disable_css_loading = True
html_static_path = ['_static']
html_css_files = ['tabs.css']

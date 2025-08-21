import os
import sys
from typing import List

sys.path.insert(0, os.path.abspath("."))


project = "portfolio-plan"
copyright = "2025, Alexandre Sonderegger"
author = "Alexandre Sonderegger"
release = "v0.1.0"


extensions = [
    "jupyter_kernel",
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns: List[str] = []


html_static_path = ["_static"]
html_theme = "pydata_sphinx_theme"
autoclass_content = "both"
nb_execution_timeout = 120

myst_enable_extensions = [
    "colon_fence",
    "attrs_block",
]
html_theme_options = {"show_toc_level": 3}
kernel_name = "portfolio_plan"

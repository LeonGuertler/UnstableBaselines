import os
import sys

# Add project root to path so autodoc can import without installing
sys.path.insert(0, os.path.abspath(".."))

project = "UnstableBaselines"
author = "Leon Guertler and contributors"
copyright = "2025, Leon Guertler"

# Keep in sync with pyproject if updated later
release = "0.2.0"

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "myst_parser",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Mock heavy optional deps to make RTD builds reliable
autodoc_mock_imports = [
    "torch",
    "vllm",
    "ray",
    "transformers",
    "peft",
    "dm_tree",
    "textarena",
    "wandb",
    "trueskill",
    "pynvml",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

napoleon_use_ivar = True
napoleon_use_admonition_for_references = True
# See https://github.com/sphinx-doc/sphinx/issues/9119
napoleon_custom_sections = [("Returns", "params_style")]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

templates_path = ["_templates"]
html_static_path = ["_static"]

html_theme = "furo"
html_logo = "_static/logo_ub.png"
html_title = "Unstable Baselines"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "black",
        "color-brand-content": "black",
        "color-announcement-background": "white",
        "color-announcement-text": "black",
        "color-sidebar-search-background": "#eeebee",
        "color-sidebar-search-background--focus": "#efeff400",
        "sidebar-caption-font-size": "87.5%",
        "color-sidebar-caption-text": "black",
    }
}

# MyST configuration for Markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

html_css_files = [
    "custom.css",
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'README.md']

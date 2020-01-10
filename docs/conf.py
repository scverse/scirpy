import sys
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from reportsrender import __author__, __version__

# General information
project = "reportsrender"
author = __author__
copyright = f"{datetime.now():%Y}, {author}."
version = __version__

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

nitpicky = True  # Warn about broken links
nitpick_ignore = [
    ("py:data", "typing.Optional"),
    ("py:class", "typing.Collection"),
    ("py:class", "typing.List"),
    ("py:class", "str"),
    ("py:class", "dict"),
]
needs_sphinx = "2.0"  # Nicer param docs


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
todo_include_todos = False


# styling
html_theme = "sphinx_rtd_theme"
pygments_style = "sphinx"
html_context = dict(
    display_github=True,  # Integrate GitHub
    github_user="grst",  # Username
    github_repo=project,  # Repo name
    github_version="master",  # Version
    conf_py_path="/docs/",  # Path in the checkout to the docs root
)

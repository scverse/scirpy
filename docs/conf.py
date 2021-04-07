import sys
from pathlib import Path
from datetime import datetime
import jupytext
from sphinx.deprecation import RemovedInSphinx40Warning
import warnings

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "extensions"))

from scirpy import __author__, __version__

# ignore Future warnings (which are caused by dependencies)
warnings.filterwarnings("ignore", category=RemovedInSphinx40Warning)
warnings.filterwarnings("ignore", category=FutureWarning)

# General information
project = "scirpy"
author = __author__
copyright = f"{datetime.now():%Y}, {author}."
version = __version__

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
api_dir = HERE / "_static" / "api"
api_rel_dir = "_static/api"
bibtex_bibfiles = ["references.bib"]


nitpicky = True  # Warn about broken links
needs_sphinx = "2.0"  # Nicer param docs


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
    "nbsphinx",
    "scanpydoc",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]


# -- auto-generate APIdoc --------------------------------------------------------------

autosummary_generate = True
autodoc_member_order = "bysource"
default_role = "literal"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False

intersphinx_mapping = dict(
    scanpy=("https://scanpy.readthedocs.io/en/stable/", None),
    anndata=("https://anndata.readthedocs.io/en/stable/", None),
    h5py=("http://docs.h5py.org/en/stable/", None),
    cycler=("https://matplotlib.org/cycler/", None),
    ipython=("https://ipython.readthedocs.io/en/stable/", None),
    leidenalg=("https://leidenalg.readthedocs.io/en/latest/", None),
    matplotlib=("https://matplotlib.org/", None),
    numpy=("https://docs.scipy.org/doc/numpy/", None),
    pandas=("https://pandas.pydata.org/pandas-docs/stable/", None),
    python=("https://docs.python.org/3", None),
    scipy=("https://docs.scipy.org/doc/scipy/reference/", None),
    seaborn=("https://seaborn.pydata.org/", None),
    sklearn=("https://scikit-learn.org/stable/", None),
    networkx=("https://networkx.github.io/documentation/networkx-1.10/", None),
    dandelion=("https://sc-dandelion.readthedocs.io/en/latest/", None),
)


# -- nbsphinx Tutorials ----------------------------------------------------------------

# Enable jupytext notebooks
nbsphinx_custom_formats = {
    ".md": lambda s: jupytext.reads(s, ".md"),
}
# nbsphinx_execute = "always"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]
nbsphinx_timeout = 300


# -- HTML styling ----------------------------------------------------------------------

html_theme = "scanpydoc"
# add custom stylesheet
# https://stackoverflow.com/a/43186995/2340703
html_static_path = ["_static"]
pygments_style = "sphinx"
html_context = dict(
    display_github=True,  # Integrate GitHub
    github_user="icbi-lab",  # Username
    github_repo=project,  # Repo name
    github_version="master",  # Version
    conf_py_path="/docs/",  # Path in the checkout to the docs root
)
html_logo = "img/scirpy_logo_bright.png"
html_theme_options = dict(navigation_depth=4, logo_only=True)


def setup(app):
    pass


# -- Supress 'reference target not found' errors ---------------------------------------

# See https://github.com/agronholm/sphinx-autodoc-typehints/issues/38 for more details.
qualname_overrides = {
    "scipy.sparse.coo.coo_matrix": "scipy.sparse.coo_matrix",
    "scirpy.io._datastructures.AirrCell": "scirpy.io.AirrCell",
    "pandas.core.arrays.categorical.Categorical": "pandas.Categorical",
    "pandas.core.series.Series": "pandas.Series",
}

nitpick_ignore = [
    ("py:class", "igraph.Graph"),
    ("py:class", "igraph.Layout"),
    ("py:class", "igraph.layout.Layout"),
    # the following entries are because the `MutableMapping` base class does not
    # use type hints.
    ("py:class", "None.  Remove all items from D."),
    ("py:class", "D[k] if k in D, else d.  d defaults to None."),
    ("py:class", "a set-like object providing a view on D's items"),
    ("py:class", "a set-like object providing a view on D's keys"),
    ("py:class", "v, remove specified key and return the corresponding value."),
    ("py:class", "(k, v), remove and return some (key, value) pair"),
    ("py:class", "D.get(k,d), also set D[k]=d if k not in D"),
    ("py:class", "None.  Update D from mapping/iterable E and F."),
    ("py:class", "an object providing a view on D's values"),
]

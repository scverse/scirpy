import os
import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

from scverse_doc.registry import intersphinx

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))

info = metadata("scirpy")
project = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}."
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
source_repository = urls["Source"]
release = info["Version"]

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]

api_dir = HERE / "_static" / "api"
api_rel_dir = "_static/api"

# Set canonical URL from the Read the Docs Domain
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

extensions = [
    "scverse_doc",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "IPython.sphinxext.ipython_console_highlighting",
    "scverse_misc.sphinx_ext",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

napoleon_use_ivar = True
napoleon_custom_sections = [("Params", "Parameters")]
myst_url_schemes = ("http", "https", "mailto")
typehints_defaults = "braces"
always_use_bars_union = True  # use `|` instead of `Union` in types even when building with Python ≤3.14

intersphinx_mapping = intersphinx("muon") | {
    "h5py": ("https://docs.h5py.org/en/stable", None),
    "cycler": ("https://matplotlib.org/cycler", None),
    "ipython": ("https://ipython.readthedocs.io/en/stable", None),
    "leidenalg": ("https://leidenalg.readthedocs.io/en/latest", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/version/2.3/", None),
    "seaborn": ("https://seaborn.pydata.org", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "networkx": ("https://networkx.org/documentation/networkx-1.10", None),
    "dandelion": ("https://sc-dandelion.readthedocs.io/en/latest", None),
    "awkward": ("https://awkward-array.org/doc/main", None),
    "pooch": ("https://www.fatiando.org/pooch/latest", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest", None),
    "logomaker": ("https://logomaker.readthedocs.io/en/latest/", None),
    "fsspec": ("https://filesystem-spec.readthedocs.io/en/stable/", None),
}

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_logo = "img/scirpy_logo.png"

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
    # Python >= 3.13 wraps the `pop` docstring, so the return type token no longer includes "value."
    ("py:class", "v, remove specified key and return the corresponding"),
    ("py:class", "(k, v), remove and return some (key, value) pair"),
    ("py:class", "D.get(k,d), also set D[k]=d if k not in D"),
    ("py:class", "None.  Update D from mapping/iterable E and F."),
    ("py:class", "an object providing a view on D's values"),
    # don't know why these are not working
    ("py:class", "seaborn.matrix.ClusterGrid"),
    ("py:meth", "mudata.MuData.update"),
    ("py:class", "awkward.highlevel.Array"),
    ("py:class", "awkward.behaviors.string.StringBehavior"),
    ("py:class", "logomaker.src.Logo.Logo"),
    ("py:class", "pathlib._local.Path"),
    ("py:data", "typing.Union"),
]

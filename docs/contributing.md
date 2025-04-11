# Contributing guide

Scanpy provides extensive [developer documentation][scanpy developer guide], most of which applies to this project, too.
This document will not reproduce the entire content from there.
Instead, it aims at summarizing the most important information to get you started on contributing.

We assume that you are already familiar with git and with making pull requests on GitHub.
If not, please refer to the [scanpy developer guide][].

## Installing dev dependencies

In addition to the packages needed to _use_ this package,
you need additional python packages to [run tests](#writing-tests) and [build the documentation](#docs-building).

The easiest way is to get familiar with [hatch environments][], with which these tasks are simply:

```bash
hatch test  # defined in the table [tool.hatch.envs.hatch-test] in pyproject.toml
hatch run docs:build  # defined in the table [tool.hatch.envs.docs]
```

[hatch environments]: https://hatch.pypa.io/latest/tutorials/environment/basic-usage/

## Code-style

This package uses [pre-commit][] to enforce consistent code-styles.
On every commit, pre-commit checks will either automatically fix issues with the code, or raise an error message.

To enable pre-commit locally, simply run

```bash
pre-commit install
```

in the root of the repository.
Pre-commit will automatically download all dependencies when it is run for the first time.

Alternatively, you can rely on the [pre-commit.ci][] service enabled on GitHub.
If you didn't run `pre-commit` before pushing changes to GitHub it will automatically commit fixes to your pull request, or show an error message.

If pre-commit.ci added a commit on a branch you still have been working on locally, simply use

```bash
git pull --rebase
```

to integrate the changes into yours.
While the [pre-commit.ci][] is useful, we strongly encourage installing and running pre-commit locally first to understand its usage.

Finally, most editors have an _autoformat on save_ feature.
Consider enabling this option for [ruff][ruff-editors] and [biome][biome-editors].


(writing-tests)=

## Writing tests

This package uses [pytest][] for automated testing.
Please write {doc}`scanpy:dev/testing` for every function added to the package.

Most IDEs integrate with pytest and provide a GUI to run tests.
Just point yours to one of the environments returned by

```bash
hatch env create hatch-test  # create test environments for all supported versions
hatch env find hatch-test  # list all possible test environment paths
```

Alternatively, you can run all tests from the command line by executing

```bash
hatch test  # test with the highest supported Python version
# or
hatch test --all  # test with all supported Python versions
```

in the root of the repository.


### Continuous integration

Continuous integration will automatically run the tests on all pull requests and test
against the minimum and maximum supported Python version.

Additionally, there's a CI job that tests against pre-releases of all dependencies
(if there are any). The purpose of this check is to detect incompatibilities
of new package versions early on and gives you time to fix the issue or reach
out to the developers of the dependency before the package is released to a wider audience.

[scanpy-test-docs]: https://scanpy.readthedocs.io/en/latest/dev/testing.html#writing-tests

## Making a release

1. **Review and update the changelog.** Make sure the changelog is up-to-date and change the top line from `[Unreleased]` to the version number you intend to publish.
2. **Create a release on GitHub.** Navigate to the "Releases" page of this project on GitHub. Specify vX.X.X as a tag name and create a release. For more information, see [managing GitHub releases][].

**That's it!** The release will be built automatically and published on PyPI using GitHub actions using the [trusted publishing workflow](https://docs.pypi.org/trusted-publishers/).
Scirpy uses [hatch-vcs](https://github.com/ofek/hatch-vcs) to automaticlly retrieve the version number from the git tag. Therefore, it is not necessary to bump versions in the code.

## Writing documentation

Please write documentation for new or changed features and use-cases. This project uses [sphinx][] with the following features:

- the [myst][] extension allows to write documentation in markdown/Markedly Structured Text
- [Numpy-style docstrings][numpydoc] (through the [napoloen][numpydoc-napoleon] extension).
- Jupyter notebooks as tutorials through [myst-nb][] (See [Tutorials with myst-nb](#tutorials-with-myst-nb-and-jupyter-notebooks))
- [Sphinx autodoc typehints][], to automatically reference annotated input and output types
- Citations (like {cite:p}`Virshup_2023`) can be included with [sphinxcontrib-bibtex](https://sphinxcontrib-bibtex.readthedocs.io/)

See the [scanpy developer docs](https://scanpy.readthedocs.io/en/latest/dev/documentation.html) for more information
on how to write documentation.

### Tutorials with myst-nb and jupyter notebooks

The documentation is set-up to render jupyter notebooks stored in the `docs/notebooks` directory using [myst-nb][].
Currently, only notebooks in `.ipynb` format are supported that will be included with both their input and output cells.
It is your responsibility to update and re-run the notebook whenever necessary.

If you are interested in automatically running notebooks as part of the continuous integration,
please check out [this feature request][issue-render-notebooks] in the `cookiecutter-scverse` repository.

[issue-render-notebooks]: https://github.com/scverse/cookiecutter-scverse/issues/40

#### Hints

- If you refer to objects from other packages, please add an entry to `intersphinx_mapping` in `docs/conf.py`.
  Only if you do so can sphinx automatically create a link to the external documentation.
- If building the documentation fails because of a missing link that is outside your control,
  you can add an entry to the `nitpick_ignore` list in `docs/conf.py`

(docs-building)=

#### Building the docs locally

```bash
hatch run docs:build
hatch run docs:open
```

<!-- Links -->

[scanpy developer guide]: https://scanpy.readthedocs.io/en/latest/dev/index.html
[cookiecutter-scverse-instance]: https://cookiecutter-scverse-instance.readthedocs.io/en/latest/template_usage.html
[github quickstart guide]: https://docs.github.com/en/get-started/quickstart/create-a-repo?tool=webui
[codecov]: https://about.codecov.io/sign-up/
[codecov docs]: https://docs.codecov.com/docs
[codecov bot]: https://docs.codecov.com/docs/team-bot
[codecov app]: https://github.com/apps/codecov
[pre-commit.ci]: https://pre-commit.ci/
[readthedocs.org]: https://readthedocs.org/
[myst-nb]: https://myst-nb.readthedocs.io/en/latest/
[jupytext]: https://jupytext.readthedocs.io/en/latest/
[pre-commit]: https://pre-commit.com/
[anndata]: https://github.com/scverse/anndata
[mudata]: https://github.com/scverse/mudata
[semver]: https://semver.org/
[sphinx]: https://www.sphinx-doc.org/en/master/
[myst]: https://myst-parser.readthedocs.io/en/latest/intro.html
[numpydoc-napoleon]: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
[numpydoc]: https://numpydoc.readthedocs.io/en/latest/format.html
[sphinx autodoc typehints]: https://github.com/tox-dev/sphinx-autodoc-typehints
[pypi]: https://pypi.org/
[managing GitHub releases]: https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository
[ruff-editors]: https://docs.astral.sh/ruff/integrations/
[biome-editors]: https://biomejs.dev/guides/integrate-in-editor/
[pytest]: https://docs.pytest.org/

# Installation

Scirpy needs Python 3.13 or newer.
We recommend [uv][], which installs Python for you and keeps each project’s dependencies in their own virtual environment.

## PyPI

Install the latest release from [PyPI][]:

:::::{tab-set}
::::{tab-item} uv
:sync: uv

```bash
uv add scirpy
```
::::

::::{tab-item} Pip
:sync: pip

```bash
pip install scirpy
```
::::
:::::

## Bioconda

First set up the conda channels [as described here](https://bioconda.github.io/#usage), then install scirpy:

```bash
conda install scirpy
```

## Development version

Install the latest development version from GitHub:

:::::{tab-set}
::::{tab-item} uv
:sync: uv

```bash
uv add git+https://github.com/scverse/scirpy.git
```
::::

::::{tab-item} Pip
:sync: pip

```bash
pip install git+https://github.com/scverse/scirpy.git
```
::::
:::::

To work on scirpy itself, see {doc}`contributing`.

## Container

Run scirpy in a container using [Docker][] or [Podman][]:

```bash
docker pull quay.io/biocontainers/scirpy:<tag>
```

where `tag` is one of [these tags](https://quay.io/repository/biocontainers/scirpy?tab=tags).

[uv]: https://docs.astral.sh/uv/
[pypi]: https://pypi.org/project/scirpy
[docker]: https://www.docker.com/
[podman]: https://podman.io/

# Installation

You need Python 3.11 or newer.
If you don't have Python installed, we recommend [Miniforge](https://github.com/conda-forge/miniforge).

## PyPI

```console
pip install scirpy
```

## Bioconda

First set up the conda channels [as described here](https://bioconda.github.io/#usage), then install scirpy:

```console
conda install scirpy
```

## Development version

```console
pip install git+https://github.com/scverse/scirpy.git@main
```

## Container

Run scirpy in a container using [Docker](https://www.docker.com/) or [Podman](https://podman.io/):

```console
docker pull quay.io/biocontainers/scirpy:<tag>
```

where `tag` is one of [these tags](https://quay.io/repository/biocontainers/scirpy?tab=tags).

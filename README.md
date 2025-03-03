# Scirpy: single-cell immune receptor analysis in Python

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]
[![PyPI][badge-pypi]][link-pypi]
[![bioconda][badge-bioconda]][link-bioconda]
[![airr][badge-airr]][link-airr]
[![Powered by NumFOCUS][badge-numfocus]][link-numfocus]

Scirpy is a package to analyse T cell receptor (TCR) or B cell receptor (BCR)
repertoires from single-cell RNA sequencing (scRNA-seq) data in Python.
It seamlessly integrates with [scanpy][] and [mudata][] and provides various modules for data import, analysis and visualization.

[//]: # "numfocus-fiscal-sponsor-attribution"

scirpy is part of the scverse® project ([website](https://scverse.org), [governance](https://scverse.org/about/roles)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).
If you like scverse® and want to support our mission, please consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
<a href="https://numfocus.org/project/scverse">
  <img
    src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
    width="200"
  >
</a>
</div>

[badge-tests]: https://img.shields.io/github/actions/workflow/status/scverse/scirpy/test.yaml?branch=main
[link-tests]: https://github.com/scverse/scirpy/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/scirpy
[badge-pypi]: https://img.shields.io/pypi/v/scirpy?logo=PyPI
[link-pypi]: https://pypi.org/project/scirpy/
[link-bioconda]: http://bioconda.github.io/recipes/scirpy/README.html
[badge-bioconda]: https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat
[badge-airr]: https://img.shields.io/static/v1?label=AIRR-C%20sw-tools%20v1&message=compliant&color=008AFF&labelColor=000000&style=flat
[link-airr]: https://docs.airr-community.org/en/stable/swtools/airr_swtools_standard.html
[scverse]: https://scverse.org
[scanpy]: https://scanpy.readthedocs.io/
[mudata]: https://github.com/scverse/mudata
[badge-numfocus]: https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A
[link-numfocus]: http://numfocus.org

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

- [Tutorial][link-tutorial], and the
- [API documentation][link-api].

## Installation

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install scirpy:

1. Install the latest release of `scirpy` from [PyPI](https://pypi.org/project/scirpy/):

    ```bash
    pip install scirpy
    ```

2. Get it from [Bioconda][link-bioconda]:

    First **setup conda channels [as described here](https://bioconda.github.io/#usage)**. Then install scirpy:

    ```bash
    conda install scirpy
    ```

3. Install the latest development version:

    ```bash
    pip install git+https://github.com/scverse/scirpy.git@main
    ```

4. Run it in a container using [Docker][] or [Podman][]:

    ```bash
    docker pull quay.io/biocontainers/scirpy:<tag>
    ```

where `tag` is one of [these tags](https://quay.io/repository/biocontainers/scirpy?tab=tags).

## Release notes

See the [changelog][changelog].

## Support and Contact

We are happy to assist with problems when using scirpy.

- If you need help with scirpy or have questions regarding single-cell immune-cell receptor analysis in general, please join us in the [scverse discourse][scverse-discourse].
- For bug report or feature requests, please use the [issue tracker][issue-tracker].

We try to respond within two working days, however fixing bugs or implementing new features
can take substantially longer, depending on the availability of our developers.

## Citation

If you use `scirpy` in your work, please cite the `scirpy`
publication as follows:

> **Scirpy: A Scanpy extension for analyzing single-cell T-cell
> receptor sequencing data**
>
> Gregor Sturm, Tamas Szabo, Georgios Fotakis, Marlene Haider, Dietmar
> Rieder, Zlatko Trajanoski, Francesca Finotello
>
> _Bioinformatics_ 2020 Sep 15. doi:
> [10.1093/bioinformatics/btaa611](https://doi.org/10.1093/bioinformatics/btaa611).

You can cite the scverse publication as follows:

> **The scverse project provides a computational ecosystem for
> single-cell omics data analysis**
>
> Isaac Virshup, Danila Bredikhin, Lukas Heumos, Giovanni Palla, Gregor
> Sturm, Adam Gayoso, Ilia Kats, Mikaela Koutrouli, Scverse Community,
> Bonnie Berger, Dana Pe’er, Aviv Regev, Sarah A. Teichmann, Francesca
> Finotello, F. Alexander Wolf, Nir Yosef, Oliver Stegle & Fabian J.
> Theis
>
> _Nat Biotechnol._ 2023 Apr 10. doi:
> [10.1038/s41587-023-01733-8](https://doi.org/10.1038/s41587-023-01733-8).

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/scverse/scirpy/issues
[changelog]: https://scirpy.readthedocs.io/latest/changelog.html
[link-docs]: https://scirpy.readthedocs.io
[link-api]: https://scirpy.readthedocs.io/latest/api.html
[link-tutorial]: https://scirpy.scverse.org/en/latest/tutorials.html
[Docker]: https://www.docker.com/
[Podman]: https://podman.io/

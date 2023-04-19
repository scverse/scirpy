"""Function to fit a negative binomial distribution. 

Adapted from https://github.com/gokceneraslan/fit_nbinom 
Copyright (C) 2014 Gokcen Eraslan
Permission granted to include it in Scirpy under BSD-License in
https://github.com/gokceneraslan/fit_nbinom/issues/5
"""

from numbers import Number
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import fmin_l_bfgs_b as optim
from scipy.special import factorial, gammaln, psi


def fit_nbinom(
    X: np.ndarray,
    initial_params: Optional[Tuple[Number, Number]] = None,
    max_iterations: Optional[int] = None,
) -> Tuple[float, float]:
    """Fit a negative binomial distribution.

    Parameters
    ----------
    X
        Data to fit. Should contain only non-negative integers.
    initial_params
        Tuple with initial `size` and `prob` parameters.
    max_iterations
        Maximum number of iterations for the optimization algorithm.

    Returns
    -------
    Tuple with the estimated `size` and `prob` parameters.
    """
    if not np.issubdtype(X.dtype, np.integer) or (X < 0).any():
        raise ValueError("Input data should contain only non-negative integers")

    infinitesimal = np.finfo(float).eps

    def negative_binomial_log_likelihood(params, *args):
        r, p = params
        X = args[0]
        N = X.size

        # MLE estimate based on the formula on Wikipedia:
        # http://en.wikipedia.org/wiki/Negative_binomial_distribution#Maximum_likelihood_estimation
        result = (
            np.sum(gammaln(X + r))
            - np.sum(np.log(factorial(X)))
            - N * (gammaln(r))
            + N * r * np.log(p)
            + np.sum(X * np.log(1 - (p if p < 1 else 1 - infinitesimal)))
        )

        return -result

    def negative_binomial_log_likelihood_deriv(params, *args):
        r, p = params
        X = args[0]
        N = X.size

        pderiv = (N * r) / p - np.sum(X) / (1 - (p if p < 1 else 1 - infinitesimal))
        rderiv = np.sum(psi(X + r)) - N * psi(r) + N * np.log(p)

        return np.array([-rderiv, -pderiv])

    if initial_params is None:
        # reasonable initial values (from fitdistr function in R)
        m = np.mean(X)
        v = np.var(X)
        size = (m**2) / (v - m) if v > m else 10

        # convert mu/size parameterization to prob/size
        p0 = size / ((size + m) if size + m != 0 else 1)
        r0 = size
        initial_params = (r0, p0)

    initial_params = np.array(initial_params)

    bounds = [(infinitesimal, None), (infinitesimal, 1)]
    optim_args = (X,)
    optim_kwargs = {"approx_grad": True, "bounds": bounds}
    if max_iterations is not None:
        optim_kwargs["maxiter"] = max_iterations
    optimres = optim(
        negative_binomial_log_likelihood,
        x0=initial_params,
        fprime=negative_binomial_log_likelihood_deriv,
        *optim_args,
        **optim_kwargs,
    )

    params = optimres[0]
    return params[0], params[1]

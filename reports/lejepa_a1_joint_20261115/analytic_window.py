"""Exact Gaussian moments for window-mean information about the cross term."""

import json

import numpy as np


def window(length, lags):
    first = np.concatenate([np.arange(length - lag) for lag in lags])
    last = np.concatenate([np.arange(lag, length) for lag in lags])
    weights = np.concatenate([
        np.full(length - lag, 1 / (len(lags) * (length - lag))) for lag in lags
    ])
    return first, last, weights


def moments(rho, length, first, last):
    time = np.arange(length)
    cov = rho ** np.abs(time[:, None] - time)
    cov_with_mean = cov.mean(axis=1)
    var_mean = cov_with_mean.mean()
    slope = cov_with_mean / var_mean
    aa = cov[np.ix_(first, first)]
    ab = cov[np.ix_(first, last)]
    ba = cov[np.ix_(last, first)]
    bb = cov[np.ix_(last, last)]
    return {
        "var_mean": var_mean,
        "plus_mean_slope": slope[last] + slope[first],
        "minus_mean_slope": slope[last] - slope[first],
        "plus_cov": aa + ab + ba + bb,
        "minus_cov": aa - ab - ba + bb,
        "plus_minus_cov": ab + bb - aa - ba,
    }


def quadratic(rho, length, lags):
    first, last, weights = window(length, lags)
    m = moments(rho, length, first, last)
    var_mean = m["var_mean"]
    var_delta = m["minus_cov"].diagonal()
    plus_slope, minus_slope = m["plus_mean_slope"], m["minus_mean_slope"]
    a = weights @ (plus_slope * (var_delta - 3 * var_mean * minus_slope**2))
    b = weights @ (plus_slope * minus_slope**2)
    explained = a*a*var_mean + 6*a*b*var_mean**2 + 15*b*b*var_mean**3
    # Wick expansion of Cov(X_i Y_i^2, X_k Y_k^2), X=sum, Y=difference.
    plus, minus, mixed = m["plus_cov"], m["minus_cov"], m["plus_minus_cov"]
    cov = plus * (np.outer(var_delta, var_delta) + 2 * minus**2) + 4 * mixed * mixed.T * minus
    total = weights @ cov @ weights
    assert 0 <= explained <= total
    return {"rho": rho, "var_window_mean": var_mean, "linear_coefficient": a,
            "cubic_coefficient": b, "population_r2_from_true_mean": explained / total}


def linear_mixing(rho_target, rho_donor, length, lags):
    first, last, weights = window(length, lags)
    target = moments(rho_target, length, first, last)
    donor = moments(rho_donor, length, first, last)
    coefficient = weights @ (target["minus_mean_slope"] * donor["minus_mean_slope"])
    explained = coefficient**2 * target["var_mean"] * donor["var_mean"]
    total = weights @ (target["minus_cov"] * donor["minus_cov"]) @ weights
    assert 0 <= explained <= total
    return {"rho_target": rho_target, "rho_donor": rho_donor,
            "population_r2_from_both_true_means": explained / total}


if __name__ == "__main__":
    # At T=2 the endpoint sum and difference are independent Gaussian variables.
    np.testing.assert_allclose(quadratic(0.5, 2, (1,))["population_r2_from_true_mean"], 1/3)
    assert linear_mixing(0.5, 0.7, 2, (1,))["population_r2_from_both_true_means"] == 0
    lags, length = (4, 16, 32), 193
    result = {
        "window_length": length,
        "lags": lags,
        "same_axis_quadratic": [quadratic(rho, length, lags) for rho in (.88, .892, .904, .906, .918, .93)],
        "linear_mixing": [linear_mixing(a, b, length, lags) for a, b in
                          ((.93, .918), (.906, .904), (.904, .88))],
    }
    print(json.dumps(result, indent=2))

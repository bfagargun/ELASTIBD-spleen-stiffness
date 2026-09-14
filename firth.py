"""Firth penalised logistic regression and small-sample helpers.

Firth's method (Firth, Biometrika 1993; Heinze & Schemper, Stat Med 2002) removes
the first-order bias of the maximum-likelihood estimator by penalising the
likelihood with the Jeffreys prior |I(beta)|^(1/2). The modified score equations
are solved by Newton-Raphson with step halving:

    U*(beta_j) = sum_i { y_i - p_i + h_i * (0.5 - p_i) } * x_ij

where h_i is the i-th diagonal element of the hat matrix
H = W^(1/2) X (X' W X)^(-1) X' W^(1/2), W = diag(p_i (1 - p_i)).

Confidence intervals are Wald intervals from the inverse information matrix
evaluated at the penalised estimate, matching the reporting in the manuscript.

Also provides the Cochran-Armitage test for trend used for the ordered
azathioprine-exposure categories.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "firth_logit",
    "logit_mle",
    "cochran_armitage",
    "wilson_ci",
]


def _design(X: pd.DataFrame | np.ndarray, add_intercept: bool = True):
    """Return the design matrix as a float array plus its column names."""
    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
        Xm = X.to_numpy(dtype=float)
    else:
        Xm = np.asarray(X, dtype=float)
        names = [f"x{i}" for i in range(Xm.shape[1])]
    if add_intercept:
        Xm = np.column_stack([np.ones(len(Xm)), Xm])
        names = ["(intercept)"] + names
    return Xm, names


def firth_logit(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    add_intercept: bool = True,
    max_iter: int = 200,
    tol: float = 1e-8,
):
    """Fit a Firth penalised logistic regression.

    Parameters
    ----------
    X : design matrix without the intercept column (a DataFrame keeps the names).
    y : binary outcome coded 0/1.

    Returns
    -------
    dict with keys ``names``, ``coef``, ``se``, ``or`` (odds ratios),
    ``ci_low``, ``ci_high``, ``p`` (Wald p-values), ``n``, ``events``,
    ``converged`` and ``table`` (a tidy DataFrame).
    """
    Xm, names = _design(X, add_intercept)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(Xm).all():
        raise ValueError("design matrix contains missing or infinite values")
    n, k = Xm.shape
    beta = np.zeros(k)
    converged = False

    def _penalised_loglik(b):
        eta = Xm @ b
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-12, None)
        info = Xm.T @ (Xm * w[:, None])
        sign, logdet = np.linalg.slogdet(info)
        ll = np.sum(y * eta - np.log1p(np.exp(eta)))
        return ll + 0.5 * logdet

    ll_old = _penalised_loglik(beta)
    for _ in range(max_iter):
        eta = Xm @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-12, None)
        info = Xm.T @ (Xm * w[:, None])
        info_inv = np.linalg.pinv(info)
        # hat-matrix diagonal
        sw = np.sqrt(w)
        Xw = Xm * sw[:, None]
        h = np.einsum("ij,jk,ik->i", Xw, info_inv, Xw)
        # modified score
        score = Xm.T @ (y - p + h * (0.5 - p))
        step = info_inv @ score
        # step halving on the penalised log-likelihood
        factor = 1.0
        for _ in range(30):
            cand = beta + factor * step
            ll_new = _penalised_loglik(cand)
            if np.isfinite(ll_new) and ll_new >= ll_old - 1e-10:
                break
            factor /= 2.0
        beta_new = beta + factor * step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            converged = True
            break
        beta, ll_old = beta_new, _penalised_loglik(beta_new)

    eta = Xm @ beta
    p = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(p * (1.0 - p), 1e-12, None)
    info = Xm.T @ (Xm * w[:, None])
    cov = np.linalg.pinv(info)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pval = 2.0 * stats.norm.sf(np.abs(z))
    out = {
        "names": names,
        "coef": beta,
        "se": se,
        "or": np.exp(beta),
        "ci_low": np.exp(beta - 1.959963984540054 * se),
        "ci_high": np.exp(beta + 1.959963984540054 * se),
        "p": pval,
        "n": int(n),
        "events": int(y.sum()),
        "converged": converged,
    }
    out["table"] = pd.DataFrame(
        {
            "term": names,
            "coef": beta,
            "se": se,
            "OR": out["or"],
            "ci_low": out["ci_low"],
            "ci_high": out["ci_high"],
            "p": pval,
        }
    )
    return out


def logit_mle(X: pd.DataFrame | np.ndarray, y: np.ndarray, add_intercept: bool = True):
    """Conventional (unpenalised) logistic regression, used for the >= 30 kPa
    outcome where the number of events is large enough."""
    import statsmodels.api as sm

    Xm, names = _design(X, add_intercept)
    model = sm.Logit(np.asarray(y, dtype=float), Xm)
    res = model.fit(disp=0, maxiter=200)
    beta = res.params
    se = res.bse
    out = {
        "names": names,
        "coef": beta,
        "se": se,
        "or": np.exp(beta),
        "ci_low": np.exp(beta - 1.959963984540054 * se),
        "ci_high": np.exp(beta + 1.959963984540054 * se),
        "p": res.pvalues,
        "n": int(len(y)),
        "events": int(np.sum(y)),
        "converged": bool(res.mle_retvals.get("converged", True)),
    }
    out["table"] = pd.DataFrame(
        {
            "term": names,
            "coef": beta,
            "se": se,
            "OR": out["or"],
            "ci_low": out["ci_low"],
            "ci_high": out["ci_high"],
            "p": out["p"],
        }
    )
    return out


def cochran_armitage(events: list[int] | np.ndarray, totals: list[int] | np.ndarray,
                     scores: list[float] | np.ndarray | None = None):
    """Cochran-Armitage test for trend across ordered categories.

    Parameters
    ----------
    events, totals : counts of the event and of all participants per category.
    scores : category scores, equally spaced (0, 1, 2, ...) by default.

    Returns
    -------
    (z, p) with a two-sided p-value.
    """
    events = np.asarray(events, dtype=float)
    totals = np.asarray(totals, dtype=float)
    if scores is None:
        scores = np.arange(len(events), dtype=float)
    scores = np.asarray(scores, dtype=float)

    N = totals.sum()
    R = events.sum()
    p_bar = R / N
    t_bar = np.sum(totals * scores) / N
    num = np.sum(scores * (events - totals * p_bar))
    var = p_bar * (1.0 - p_bar) * np.sum(totals * (scores - t_bar) ** 2)
    if var <= 0:
        return np.nan, np.nan
    z = num / np.sqrt(var)
    return z, 2.0 * stats.norm.sf(abs(z))


def wilson_ci(k: int, n: int, alpha: float = 0.05):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.isf(alpha / 2.0)
    phat = k / n
    denom = 1.0 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))

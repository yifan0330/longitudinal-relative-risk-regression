"""Penalized logistic GEE routines ported from PGEE_source.R."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import t as student_t

_STRICT_ORIGINAL = False


def _pinv(a):
    a = np.asarray(a, dtype=float)
    if _STRICT_ORIGINAL:
        _require_finite(a, "matrix inverse input")
        return np.linalg.inv(a)
    return np.linalg.pinv(a)


def _require_finite(values, label: str) -> None:
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(f"Non-finite {label}")


def _variance(values):
    values = np.asarray(values, dtype=float)
    if _STRICT_ORIGINAL:
        _require_finite(values, "variance vector")
        if np.any(values <= 0.0):
            raise FloatingPointError("Non-positive variance vector")
        return values
    return np.clip(values, 1e-12, None)


def _sqrt_diag(values, label: str):
    diag = np.diag(values)
    if _STRICT_ORIGINAL:
        _require_finite(diag, label)
        if np.any(diag < 0.0):
            raise FloatingPointError(f"Negative {label}")
        return np.sqrt(diag)
    return np.sqrt(np.clip(diag, 0, None))


def xch(n: int, alpha: float) -> np.ndarray:
    r = np.full((int(n), int(n)), float(alpha))
    np.fill_diagonal(r, 1.0)
    return r


def ar1(n: int, alpha: float) -> np.ndarray:
    idx = np.arange(int(n))
    return float(alpha) ** np.abs(idx[:, None] - idx[None, :])


def W_f(var_mu):
    return [np.diag(np.asarray(v, dtype=float).reshape(-1)) for v in var_mu]


def error(y, mu, var_mu):
    return [
        (np.asarray(yi) - np.asarray(mi)) / np.sqrt(_variance(vi))
        for yi, mi, vi in zip(y, mu, var_mu)
    ]


def AR1_f(e):
    out = []
    alphas = []
    for ei in e:
        ei = np.asarray(ei, dtype=float)
        if len(ei) > 1:
            alphas.append(np.mean(ei[:-1] * ei[1:]))
    alpha = float(np.mean(alphas)) if alphas else 0.0
    alpha = np.clip(alpha, -0.95, 0.95)
    for ei in e:
        out.append(ar1(len(ei), alpha))
    return out


def R_exch(e):
    alphas = []
    out = []
    for ei in e:
        ei = np.asarray(ei, dtype=float)
        ni = len(ei)
        if ni > 1:
            p = np.outer(ei, ei)
            alphas.append((p.sum() - np.trace(p)) / (ni * (ni - 1)))
    alpha = float(np.mean(alphas)) if alphas else 0.0
    alpha = np.clip(alpha, -0.95, 0.95)
    for ei in e:
        out.append(xch(len(ei), alpha))
    return out


def phi_f(e):
    return float(np.mean([np.mean(np.asarray(x) ** 2) for x in e]))


def pV_f(W, R, phi):
    vals = []
    for Wi, Ri in zip(W, R):
        w12 = np.diag(_sqrt_diag(Wi, "working variance"))
        vals.append(phi * w12 @ Ri @ w12)
    return vals


def pD_f(W, x):
    return [np.asarray(xi, dtype=float).T @ Wi for Wi, xi in zip(W, x)]


def d_f(x, W, V, y, mu):
    return [
        np.asarray(xi).T @ Wi @ _pinv(Vi) @ (np.asarray(yi) - np.asarray(mi))
        for xi, Wi, Vi, yi, mi in zip(x, W, V, y, mu)
    ]


def Vmi_f(D, V):
    total = sum(Di @ _pinv(Vi) @ Di.T for Di, Vi in zip(D, V))
    return _pinv(total)


def Bi_f(D, di, V):
    n_clusters = len(D)
    p = D[0].shape[0]
    d_bar = sum(di) / n_clusters
    B1 = sum(np.outer(dii - d_bar, dii - d_bar) for dii in di)
    n = sum(Vi.shape[0] for Vi in V)
    denom = max(n - p, 1)
    return B1 * (n_clusters / max(n_clusters - 1, 1)) * ((n - 1) / denom)


def Vs_f(Vm, B):
    return Vm @ B @ Vm


def VsM_f(Vm, B, nc):
    fi = max(1.0, np.trace(Vm @ B) / B.shape[0])
    deln = min(0.5, B.shape[0] / max(nc - B.shape[0], 1))
    out = Vm @ B @ Vm + fi * deln * Vm
    if _STRICT_ORIGINAL:
        _require_finite(out, "modified sandwich variance")
        if np.any(np.diag(out) < 0.0):
            raise FloatingPointError("Negative modified sandwich variance")
    else:
        np.fill_diagonal(out, np.abs(np.diag(out)))
    return out


def Ii_f(x, W, R, phi):
    parts = []
    for xi, Wi, Ri in zip(x, W, R):
        xi = np.asarray(xi, dtype=float)
        w12 = np.diag(_sqrt_diag(Wi, "working variance"))
        parts.append(xi.T @ w12 @ _pinv(Ri) @ w12 @ xi)
    return sum(parts) / max(phi, 1e-12)


def Q_f(mu):
    return [np.diag(0.5 - np.asarray(m).reshape(-1)) for m in mu]


def Z_f(x):
    return [
        [np.diag(np.asarray(xi)[:, j]) for j in range(np.asarray(xi).shape[1])]
        for xi in x
    ]


def FFi_f(x, W, R, Q, Z, phi):
    p = np.asarray(x[0]).shape[1]
    FF = [np.zeros((p, p)) for _ in range(p)]
    for xi, Wi, Ri, Qi, Zi in zip(x, W, R, Q, Z):
        xi = np.asarray(xi, dtype=float)
        w12 = np.diag(_sqrt_diag(Wi, "working variance"))
        base = xi.T @ w12 @ _pinv(Ri) @ w12 @ Qi
        for j in range(p):
            FF[j] += base @ Zi[j] @ xi
    return [[2.0 * mat / max(phi, 1e-12) for mat in FF]]


def Pi_f(I, FF):
    I_inv = _pinv(I)
    return np.asarray([np.trace(I_inv @ f) for f in FF[0]])


def Ustari_f(x, W, V, y, mu, P):
    U = sum(
        np.asarray(xi).T @ Wi @ _pinv(Vi) @ (np.asarray(yi) - np.asarray(mi))
        for xi, Wi, Vi, yi, mi in zip(x, W, V, y, mu)
    )
    return U + 0.5 * np.asarray(P)


def _split_by_id(frame: pd.DataFrame, ids, columns):
    frame = frame.copy()
    frame["__id__"] = np.asarray(ids)
    return [
        g[columns].to_numpy(dtype=float) for _, g in frame.groupby("__id__", sort=False)
    ]


def GEE_sd_corr(y, x, beta, nc, ar=False):
    z = [np.asarray(xi) @ beta for xi in x]
    mu = [expit(zi) for zi in z]
    var_mu = [_variance(mi * (1 - mi)) for mi in mu]
    W = W_f(var_mu)
    e = error(y, mu, var_mu)
    R = AR1_f(e) if ar else R_exch(e)
    phi = phi_f(e)
    V = pV_f(W, R, phi)
    D = pD_f(W, x)
    Vm = Vmi_f(D, V)
    di = d_f(x, W, V, y, mu)
    B = Bi_f(D, di, V)
    Vs = Vs_f(Vm, B)
    VsM = VsM_f(Vm, B, nc)
    corr = R[0][1, 0] if R and R[0].shape[0] > 1 else 0.0
    return {
        "Vs": _sqrt_diag(Vs, "sandwich variance"),
        "VsMod": _sqrt_diag(VsM, "modified sandwich variance"),
        "corr": np.repeat(corr, len(beta)),
    }


def _geefirth_impl(
    y,
    x,
    ids,
    ar=True,
    init_offset=0.0,
    keep_trace=False,
    penalized=True,
    max_iter=25,
    tol=1e-4,
    strict_original=False,
):
    global _STRICT_ORIGINAL
    old_strict = _STRICT_ORIGINAL
    _STRICT_ORIGINAL = strict_original
    try:
        return _geefirth_impl_inner(
            y,
            x,
            ids,
            ar=ar,
            init_offset=init_offset,
            keep_trace=keep_trace,
            penalized=penalized,
            max_iter=max_iter,
            tol=tol,
        )
    finally:
        _STRICT_ORIGINAL = old_strict


def _geefirth_impl_inner(
    y,
    x,
    ids,
    ar=True,
    init_offset=0.0,
    keep_trace=False,
    penalized=True,
    max_iter=25,
    tol=1e-4,
):
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    x_df = pd.DataFrame(x).reset_index(drop=True)
    ids_arr = np.asarray(ids)
    df = pd.concat([pd.Series(y_arr, name="y"), x_df], axis=1).dropna()
    if df.empty:
        raise ValueError("No complete observations")
    id_counts = pd.Series(ids_arr[df.index]).value_counts()
    single = id_counts[id_counts == 1]
    if len(single):
        raise ValueError(
            f"Clusters with a single observation: {','.join(map(str, single.index))}"
        )

    X_df = pd.concat(
        [pd.Series(1.0, index=x_df.index, name="intercept"), x_df], axis=1
    ).iloc[df.index]
    names = list(X_df.columns)
    beta = np.r_[np.log(np.mean(df["y"]) + init_offset), np.zeros(X_df.shape[1] - 1)]
    x_split = _split_by_id(X_df, ids_arr[df.index], names)
    y_split = [
        g["y"].to_numpy(dtype=float)
        for _, g in pd.concat(
            [df[["y"]], pd.Series(ids_arr[df.index], name="__id__")], axis=1
        ).groupby("__id__", sort=False)
    ]
    nc = len(y_split)
    se_model_trace = []
    se_model = np.full(len(beta), np.nan)
    phi = np.nan

    for counter in range(1, max_iter + 1):
        mu = [expit(np.asarray(xi) @ beta) for xi in x_split]
        var_mu = [_variance(m * (1 - m)) for m in mu]
        W = W_f(var_mu)
        e = error(y_split, mu, var_mu)
        R = AR1_f(e) if ar else R_exch(e)
        phi = phi_f(e)
        V = pV_f(W, R, phi)
        I = Ii_f(x_split, W, R, phi)
        P = 0.0
        if penalized:
            Q = Q_f(mu)
            Z = Z_f(x_split)
            FF = FFi_f(x_split, W, R, Q, Z, phi)
            P = Pi_f(I, FF)
        Ustar = Ustari_f(x_split, W, V, y_split, mu, P)
        step = _pinv(I) @ Ustar
        beta = beta + step
        _require_finite(beta, "coefficient update")
        se_model = _sqrt_diag(_pinv(I), "model variance")
        se_model_trace.append(se_model.copy())
        if np.max(np.abs(step)) <= tol:
            break

    sd = GEE_sd_corr(y_split, x_split, beta, nc, ar)
    n = len(df)
    p = len(beta)
    t_sw = beta / sd["Vs"]
    t_swm = beta / sd["VsMod"]
    p_sw = student_t.sf(np.abs(t_sw), max(n - p, 1))
    p_swm = student_t.sf(np.abs(t_swm), max(n - p, 1))
    est_sw = pd.DataFrame(
        {
            "coefficients": beta,
            "std.err": sd["Vs"],
            "Wald": np.round(t_sw, 4) ** 2,
            "p.val": np.round(p_sw, 4),
        },
        index=names,
    )
    est_swm = pd.DataFrame(
        {
            "coefficients": beta,
            "std.err": sd["VsMod"],
            "Wald": np.round(t_swm, 4) ** 2,
            "p.val": np.round(p_swm, 4),
        },
        index=names,
    )
    result = [
        est_sw,
        est_swm,
        sd["corr"][1] if len(sd["corr"]) > 1 else sd["corr"][0],
        counter,
        phi,
    ]
    if keep_trace:
        result.extend([se_model, np.asarray(se_model_trace)])
    return result


def geefirth(
    y,
    x,
    id,
    ar=True,
    max_iter=25,
    tol=1e-4,
    strict_original=False,
    penalized=True,
):
    return _geefirth_impl(
        y,
        x,
        id,
        ar=ar,
        init_offset=0.0,
        keep_trace=False,
        penalized=penalized,
        max_iter=max_iter,
        tol=tol,
        strict_original=strict_original,
    )

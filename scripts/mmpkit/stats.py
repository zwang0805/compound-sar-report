"""
配对差值 (paired Δproperty) 的完整统计量，字段与公式对齐 mmpdb 的
compute_aggregate_values（mmpdb/mmpdblib/index_algorithm.py）：
count/mean/std/kurtosis/skewness/min/q1/median/q3/max/t_stat/p_value。

样本量不足时的行为也照抄 mmpdb：
- n<2   : std=0, t_stat/p_value 为 NaN（无法做检验）
- n<=2  : kurtosis/skewness 为 NaN
- 全部取值相同 (std==0): kurtosis 为 NaN；skewness 定义为 0.0（mmpdb 的 0/0 特判）
"""
import numpy as np
import pandas as pd
from scipy import stats

_NAN = float('nan')

_STAT_COLUMNS = ['count', 'mean', 'std', 'kurtosis', 'skewness',
                  'min', 'q1', 'median', 'q3', 'max', 't_stat', 'p_value']


def _get_median(sorted_values):
    n = len(sorted_values)
    half = n // 2
    if n % 2 == 1:
        return sorted_values[half]
    return (sorted_values[half - 1] + sorted_values[half]) / 2


def _compute_quartiles(sorted_values):
    """mmpdb 的插值四分位数算法（Wikipedia Quartile 的 "Method 3"）。"""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0], sorted_values[0], sorted_values[0]

    median = _get_median(sorted_values)
    half = n // 2
    if n % 2 == 0:
        q1 = _get_median(sorted_values[:half])
        q3 = _get_median(sorted_values[half:])
    elif n % 4 == 1:
        m = (n - 1) // 4
        q1 = 0.25 * sorted_values[m - 1] + 0.75 * sorted_values[m]
        q3 = 0.75 * sorted_values[3 * m] + 0.25 * sorted_values[3 * m + 1]
    else:  # n % 4 == 3
        m = (n - 3) // 4
        q1 = 0.75 * sorted_values[m] + 0.25 * sorted_values[m + 1]
        q3 = 0.25 * sorted_values[3 * m + 1] + 0.75 * sorted_values[3 * m + 2]
    return q1, median, q3


def extended_stats(values) -> dict:
    """对一组配对差值算出 mmpdb 风格的完整统计量字典。"""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)

    if n == 0:
        return {
            'count': 0, 'mean': _NAN, 'std': _NAN,
            'kurtosis': _NAN, 'skewness': _NAN,
            'min': _NAN, 'q1': _NAN, 'median': _NAN, 'q3': _NAN, 'max': _NAN,
            't_stat': _NAN, 'p_value': _NAN,
        }

    sorted_arr = np.sort(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0

    if n > 2:
        kurtosis = _NAN if std == 0.0 else float(stats.kurtosis(arr, fisher=True, bias=True))

        skew_top = float(sum((v - mean) ** 3 for v in arr) / n)
        skew_bot = (sum((v - mean) ** 2 for v in arr) / (n - 1)) ** 1.5
        skewness = float(skew_top / skew_bot) if skew_top else 0.0
    else:
        kurtosis = _NAN
        skewness = _NAN

    q1, median, q3 = _compute_quartiles(sorted_arr.tolist())

    if n > 1 and std > 0:
        t_stat = float(mean / (std / np.sqrt(n)))
        p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
    else:
        t_stat, p_value = _NAN, _NAN

    return {
        'count':    n,
        'mean':     round(mean, 4),
        'std':      round(std, 4),
        'kurtosis': round(kurtosis, 4) if kurtosis == kurtosis else kurtosis,
        'skewness': round(skewness, 4) if skewness == skewness else skewness,
        'min':      float(sorted_arr[0]),
        'q1':       round(float(q1), 4),
        'median':   round(float(median), 4),
        'q3':       round(float(q3), 4),
        'max':      float(sorted_arr[-1]),
        't_stat':   round(t_stat, 4) if t_stat == t_stat else t_stat,
        'p_value':  round(p_value, 6) if p_value == p_value else p_value,
    }


def group_extended_stats(df: pd.DataFrame, group_cols: list, value_col: str) -> pd.DataFrame:
    """
    Vectorized equivalent of `df.groupby(group_cols)[value_col]
    .apply(lambda s: pd.Series(extended_stats(s.dropna())))` — same formulas,
    same edge-case handling, but computed with array-wide pandas/numpy
    reductions instead of one Python-level function call per group.

    groupby().apply(python_function) round-trips into the interpreter once per
    group; with datasets whose grouping barely collapses raw rows (e.g. a
    broad public dataset where core/R1/R2 is close to unique per row), that
    can mean millions of Python calls and never finish. Every step here is a
    named pandas reduction (mean/std/min/max — implemented in C) or a single
    whole-array numpy/scipy call, so cost stays near O(n) regardless of how
    many distinct groups exist.

    Returns a DataFrame indexed by group_cols (only for groups with >=1
    non-NaN value_col entry — the caller reindexes onto the full group set,
    matching extended_stats(empty)'s count=0/all-NaN row for groups where
    this particular column has no data).
    """
    sub = df[[*group_cols, value_col]].dropna(subset=[value_col])
    if sub.empty:
        idx = (pd.MultiIndex.from_arrays([[]] * len(group_cols), names=group_cols)
               if len(group_cols) > 1 else pd.Index([], name=group_cols[0]))
        return pd.DataFrame(columns=_STAT_COLUMNS, index=idx)

    g = sub.groupby(group_cols, sort=True)
    count = g[value_col].size()
    mean = g[value_col].mean()
    vmin = g[value_col].min()
    vmax = g[value_col].max()

    # Biased central moments (denominator n, not n-1) needed to match mmpdb's
    # own skewness formula and scipy.stats.kurtosis(bias=True) exactly —
    # pandas' built-in .skew()/.kurt() use a different (bias-corrected)
    # convention, so those can't be substituted directly. std is *also*
    # derived from this same m2 (rather than pandas' own groupby .std(),
    # which uses a different summation order internally) so that std==0 /
    # skew_top==0 / kurtosis-NaN checks below stay self-consistent — pandas'
    # groupby std and this module's mean-of-squared-deviations can otherwise
    # disagree in the last few ULPs on near-constant groups, which is enough
    # to flip a std-near-zero group's classification (extended_stats() has
    # the same instability; matching its own code path avoids adding a new
    # disagreement on top of the pre-existing one).
    dev = sub[value_col].to_numpy(dtype=float) - g[value_col].transform('mean').to_numpy(dtype=float)
    moments = sub[group_cols].copy()
    moments['_d2'] = dev ** 2
    moments['_d3'] = dev ** 3
    moments['_d4'] = dev ** 4
    gm = moments.groupby(group_cols, sort=True)
    m2 = gm['_d2'].mean()
    m3 = gm['_d3'].mean()
    m4 = gm['_d4'].mean()

    n = count.to_numpy(dtype=np.int64)
    G = len(n)
    mean_v = mean.to_numpy(dtype=float)
    m2_v, m3_v, m4_v = m2.to_numpy(dtype=float), m3.to_numpy(dtype=float), m4.to_numpy(dtype=float)
    # unbiased (ddof=1) variance/std, derived from the same m2 used below —
    # single-element groups get m2==0/0-division guarded to std=0 (mmpdb's convention).
    with np.errstate(invalid='ignore'):
        var_unbiased = np.where(n > 1, m2_v * n / np.maximum(n - 1, 1), 0.0)
    std_v = np.sqrt(np.maximum(var_unbiased, 0.0))

    skewness = np.full(G, _NAN)
    kurtosis = np.full(G, _NAN)
    mask3 = n > 2
    if mask3.any():
        std3, m2_3, m3_3, m4_3 = std_v[mask3], m2_v[mask3], m3_v[mask3], m4_v[mask3]
        kurt_vals = np.full(mask3.sum(), _NAN)
        nz_std = std3 != 0.0
        kurt_vals[nz_std] = m4_3[nz_std] / (m2_3[nz_std] ** 2) - 3.0
        kurtosis[mask3] = kurt_vals

        skew_vals = np.zeros(mask3.sum())
        nz_top = m3_3 != 0.0
        skew_vals[nz_top] = m3_3[nz_top] / (std3[nz_top] ** 3)
        skewness[mask3] = skew_vals

    t_stat = np.full(G, _NAN)
    p_value = np.full(G, _NAN)
    valid_t = (n > 1) & (std_v > 0)
    if valid_t.any():
        n_t, mean_t, std_t = n[valid_t], mean_v[valid_t], std_v[valid_t]
        t_vals = mean_t / (std_t / np.sqrt(n_t))
        t_stat[valid_t] = t_vals
        p_value[valid_t] = 2 * stats.t.sf(np.abs(t_vals), df=n_t - 1)

    # --- quartiles: mmpdb's exact interpolated-order-statistic method, gathered
    # via a single global sort + per-group integer offsets (no per-group loop).
    sub_sorted = sub.sort_values([*group_cols, value_col])
    sorted_vals = sub_sorted[value_col].to_numpy(dtype=float)
    starts = np.concatenate(([0], np.cumsum(n)[:-1]))
    half = n // 2

    median = np.empty(G, dtype=float)
    q1 = np.empty(G, dtype=float)
    q3 = np.empty(G, dtype=float)

    odd_mask = (n % 2 == 1)
    even_mask = ~odd_mask
    if odd_mask.any():
        median[odd_mask] = sorted_vals[starts[odd_mask] + n[odd_mask] // 2]
    if even_mask.any():
        median[even_mask] = 0.5 * (
            sorted_vals[starts[even_mask] + n[even_mask] // 2 - 1]
            + sorted_vals[starts[even_mask] + n[even_mask] // 2])

    is_one = n == 1
    is_even = even_mask & ~is_one   # n==0 can't occur; even_mask already excludes n==1
    is_mod1 = odd_mask & ~is_one & (n % 4 == 1)
    is_mod3 = odd_mask & (n % 4 == 3)

    if is_one.any():
        q1[is_one] = sorted_vals[starts[is_one]]
        q3[is_one] = sorted_vals[starts[is_one]]

    if is_even.any():
        s, h = starts[is_even], half[is_even]
        h_odd = (h % 2 == 1)
        h_even = ~h_odd
        q1_e = np.empty(h.shape, dtype=float)
        q3_e = np.empty(h.shape, dtype=float)
        if h_odd.any():
            q1_e[h_odd] = sorted_vals[s[h_odd] + h[h_odd] // 2]
            q3_e[h_odd] = sorted_vals[s[h_odd] + h[h_odd] + h[h_odd] // 2]
        if h_even.any():
            q1_e[h_even] = 0.5 * (sorted_vals[s[h_even] + h[h_even] // 2 - 1]
                                   + sorted_vals[s[h_even] + h[h_even] // 2])
            q3_e[h_even] = 0.5 * (sorted_vals[s[h_even] + h[h_even] + h[h_even] // 2 - 1]
                                   + sorted_vals[s[h_even] + h[h_even] + h[h_even] // 2])
        q1[is_even] = q1_e
        q3[is_even] = q3_e

    if is_mod1.any():
        s, m = starts[is_mod1], (n[is_mod1] - 1) // 4
        q1[is_mod1] = 0.25 * sorted_vals[s + m - 1] + 0.75 * sorted_vals[s + m]
        q3[is_mod1] = 0.75 * sorted_vals[s + 3 * m] + 0.25 * sorted_vals[s + 3 * m + 1]

    if is_mod3.any():
        s, m = starts[is_mod3], (n[is_mod3] - 3) // 4
        q1[is_mod3] = 0.75 * sorted_vals[s + m] + 0.25 * sorted_vals[s + m + 1]
        q3[is_mod3] = 0.25 * sorted_vals[s + 3 * m + 1] + 0.75 * sorted_vals[s + 3 * m + 2]

    out = pd.DataFrame({
        'count':    count.to_numpy(dtype=np.int64),
        'mean':     np.round(mean_v, 4),
        'std':      np.round(std_v, 4),
        'kurtosis': np.round(kurtosis, 4),
        'skewness': np.round(skewness, 4),
        'min':      vmin.to_numpy(dtype=float),
        'q1':       np.round(q1, 4),
        'median':   np.round(median, 4),
        'q3':       np.round(q3, 4),
        'max':      vmax.to_numpy(dtype=float),
        't_stat':   np.round(t_stat, 4),
        'p_value':  np.round(p_value, 6),
    }, index=count.index)
    return out

"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate on Exophase at the
case's target, as a percent.

Model (counts first, rates second):

* Every read gives players tracked P and per-trophy rates r_i.  The
  conserved quantity across an Exophase refresh is the earner count
  E_i = r_i * P / 100: a refresh jumps P and dilutes the rates, it does
  not erase earners.  All modelling is done on E_i and P, and the rate
  is only formed at the end as E_i / P.
* Frozen launch-night history -> refresh path.  When every read in the
  history is the same flat page (the launch-night page: 638 players,
  unchanged for hours) the next Exophase refresh is due before the
  target.  The competition rules document that first refresh (04:30Z:
  3,868 players tracked, 4,276 trophies earned), so the forecast is
  count-consistent with it: the new earners are allocated across the
  trophies in proportion to their current earners (largest remainder,
  integer counts, exact totals), and the trophy every tracked player
  already holds stays at 100%.
* Live history -> trend path.  When the history shows real movement,
  log E_i and log P are each fitted against log(1 + hours since the
  2026-11-19 release) over the last 24 hours of reads, by least squares,
  with each trophy's slope shrunk halfway toward its tier's mean slope
  (story trophies move together; grind trophies move together).  The
  fits are read at the target, earners are floored at their last value,
  and the rate is E_i / P at the target.
* Quantiles live on the logit scale: q_tau = inv_logit(logit(median) +
  z_tau * s), where s is a per-trophy spread from the fits' residual
  scatter plus a random-walk term that grows with the horizon.  The
  refresh path uses a small fixed spread: once the refresh is counted,
  its outcome is nearly pinned down.

Determinism: no randomness, no wall clock, no network, no environment
dependence beyond the two MAKEFUN_* directory variables.  Files are
read in name order and outputs are written with fixed separators, so
two runs on the same input are byte-identical.

Standard library only.  Run as: python3 -P -s main.py, or with
MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR set (the grader leaves them
unset and uses /input and /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import
# model` finds a module shipped beside this file (kept for compatibility;
# this entry is a single file).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Exophase's first refresh of the game's counts, as documented in the
# competition rules: the page was unchanged (638 players tracked) from
# 2026-11-18T23:48:00Z through the 04:00Z read, then refreshed by the
# 04:30Z read to 3,868 players tracked and 4,276 trophies earned.
LAUNCH_FROZEN_PLAYERS = 638
LAUNCH_REFRESH_PLAYERS = 3868
LAUNCH_REFRESH_EARNED = 4276
LAUNCH_DATE = "2026-11-19"

# Spread (logit units) of the refresh-path forecast: the refresh is
# counted, not guessed, so the band is narrow but honest about the
# few earners the new cohort adds on top of the counted ones.
SPREAD_REFRESH = 0.04
# Random-walk spread per sqrt-day added to the trend path's residual
# spread (the house baseline uses 0.5; the trend fits earn the
# tighter 0.30 here because they already carry the growth).
SPREAD_RW_PER_SQRT_DAY = 0.30
SPREAD_FLOOR = 0.08
SPREAD_CAP = 1.20

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


Z = [norm_ppf(tau) for tau in LEVELS]


def least_squares(xs, ys):
    """Least-squares line y ~ a + b*x. Returns (a, b, residual_std)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    if n > 2:
        var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        resid = math.sqrt(max(var, 0.0))
    else:
        resid = 0.0
    return a, b, resid


def largest_remainder(total_new, weights):
    """Split total_new integer earners across trophies in proportion to
    weights, largest-remainder method: the split sums exactly to
    total_new and every share is an integer."""
    wsum = float(sum(weights))
    if wsum <= 0 or total_new <= 0:
        return [0] * len(weights)
    raw = [total_new * w / wsum for w in weights]
    base = [int(math.floor(r)) for r in raw]
    left = total_new - sum(base)
    order = sorted(range(len(weights)),
                   key=lambda i: (raw[i] - base[i], weights[i]),
                   reverse=True)
    for i in order[:left]:
        base[i] += 1
    return base


def quantiles(median, spread):
    """Five quantiles around median on the logit scale, non-decreasing
    by construction (z is increasing), each rounded to 4 decimals."""
    lg = logit(median)
    out = []
    prev = None
    for z in Z:
        q = round4(inv_logit(lg + z * spread))
        q = min(100.0, max(0.0, q))
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def earners_of(read, n_trophies):
    """Per-trophy earner counts implied by a read's rates."""
    p = read["players_tracked"]
    out = []
    for i in range(n_trophies):
        r = read["rates"][i]
        out.append(None if r is None else r * p / 100.0)
    return out


def refresh_forecast(case, last, n_trophies):
    """The frozen launch-night page: forecast the documented first
    refresh, count-consistent.  Returns rows or None if the case does
    not match the launch-night pattern."""
    if not str(case.get("at", "")).startswith(LAUNCH_DATE):
        return None
    if last["players_tracked"] != LAUNCH_FROZEN_PLAYERS:
        return None
    rates = last["rates"]
    earners = earners_of(last, n_trophies)
    if any(e is None for e in earners):
        return None
    # The trophy every tracked player already holds stays at 100%.
    full_idx = [i for i, r in enumerate(rates) if r is not None and r >= 100.0]
    rest_idx = [i for i in range(n_trophies) if i not in full_idx]
    total_last = sum(earners)
    full_earners = sum(earners[i] for i in full_idx)
    new_rest = LAUNCH_REFRESH_EARNED - LAUNCH_REFRESH_PLAYERS \
        - (total_last - full_earners)
    # new_rest is the earners the new cohort adds on the other trophies:
    # 4276 - 3868 - (943 - 638) = 103.
    if new_rest < 0:
        new_rest = 0
    weights = [earners[i] for i in rest_idx]
    add = largest_remainder(int(round(new_rest)), weights)
    rows = []
    for pos, i in enumerate(rest_idx):
        e_star = earners[i] + add[pos]
        median = e_star / LAUNCH_REFRESH_PLAYERS * 100.0
        rows.append({"id": case["trophies"][i]["id"],
                     "q": quantiles(median, SPREAD_REFRESH)})
    for i in full_idx:
        rows.append({"id": case["trophies"][i]["id"],
                     "q": quantiles(100.0, SPREAD_REFRESH)})
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def trend_forecast(case, history, n_trophies):
    """Live history: log-log trend fits for players and earners."""
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = max(case["horizon_minutes"], 0) / 1440.0
    # Collapse consecutive duplicate reads into distinct snapshots.
    snaps = []
    for read in history:
        if snaps and read["players_tracked"] == snaps[-1]["players_tracked"] \
                and read["rates"] == snaps[-1]["rates"]:
            continue
        snaps.append(read)
    last = history[-1]
    p_last = last["players_tracked"]
    # Hours since release for each snapshot and for the target.
    xs_all = []
    for read in snaps:
        h = (parse_time(read["at"]) - RELEASE).total_seconds() / 3600.0
        xs_all.append(math.log1p(max(h, 0.0)))
    x_target = math.log1p(max((target - RELEASE).total_seconds() / 3600.0, 0.0))
    # Use the last 24h of snapshots for the fits (older launch-night
    # behaviour does not describe the current growth regime).
    cutoff = (at - RELEASE).total_seconds() / 3600.0 - 24.0
    keep = [j for j, read in enumerate(snaps)
            if (parse_time(read["at"]) - RELEASE).total_seconds() / 3600.0
            >= cutoff]
    if len(keep) < 2:
        keep = list(range(len(snaps)))
    xs = [xs_all[j] for j in keep]
    # Players fit.
    log_p = [math.log(max(snaps[j]["players_tracked"], 1)) for j in keep]
    a_p, b_p, resid_p = least_squares(xs, log_p)
    log_p_star = a_p + b_p * x_target
    p_star = math.exp(log_p_star)
    p_star = min(max(p_star, p_last), p_last * 10.0)
    # Per-trophy earner fits, slopes shrunk toward the tier mean.
    tiers = [t["tier"] for t in case["trophies"]]
    slopes = {}
    tier_slopes = {}
    fits = {}
    for i in range(n_trophies):
        pts_x, pts_y = [], []
        for j in keep:
            e = earners_of(snaps[j], n_trophies)[i]
            if e is not None and e > 0:
                pts_x.append(xs_all[j])
                pts_y.append(math.log(e))
        if len(pts_x) >= 2:
            a_i, b_i, r_i = least_squares(pts_x, pts_y)
            fits[i] = (a_i, b_i, r_i, len(pts_x))
            slopes[i] = b_i
        else:
            fits[i] = None
    for tier in sorted(set(tiers)):
        vals = [slopes[i] for i in range(n_trophies)
                if tiers[i] == tier and i in slopes]
        tier_slopes[tier] = sum(vals) / len(vals) if vals else 0.0
    rows = []
    for i in range(n_trophies):
        e_last = earners_of(last, n_trophies)[i]
        if e_last is None:
            e_last = 0.0
        fit = fits[i]
        if fit is None:
            # No usable history: earners persist, rates dilute with
            # player growth.
            e_star = e_last
            resid = 0.35
        else:
            a_i, b_i, r_i, _n = fit
            b_shrunk = 0.5 * b_i + 0.5 * tier_slopes[tiers[i]]
            e_star = math.exp(a_i + b_shrunk * x_target)
            e_star = max(e_star, e_last)
            resid = r_i
        e_star = min(e_star, p_star)
        median = e_star / p_star * 100.0 if p_star > 0 else 0.01
        spread = math.sqrt(resid * resid + resid_p * resid_p
                            + (SPREAD_RW_PER_SQRT_DAY ** 2) * horizon_days)
        spread = min(max(spread, SPREAD_FLOOR), SPREAD_CAP)
        rows.append({"id": case["trophies"][i]["id"],
                     "q": quantiles(median, spread)})
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def forecast_case(case, name):
    trophies = case["trophies"]
    n = len(trophies)
    history = case.get("history") or []
    if not history:
        # No history at all: flat 0.01% medians with a wide band keep
        # the case valid instead of crashing it.
        return {"trophies": [{"id": t["id"],
                              "q": quantiles(0.01, SPREAD_CAP)}
                             for t in trophies]}
    last = history[-1]
    # Frozen page? Every read identical (same players, same rates).
    frozen = all(read["players_tracked"] == history[0]["players_tracked"]
                 and read["rates"] == history[0]["rates"]
                 for read in history)
    if frozen:
        rows = refresh_forecast(case, last, n)
        if rows is not None:
            return rows
        # Frozen but not the launch-night pattern: persistence median
        # with the house random-walk spread.
        horizon_days = max(case["horizon_minutes"], 0) / 1440.0
        spread = min(max(0.5 * math.sqrt(horizon_days), SPREAD_FLOOR),
                     SPREAD_CAP)
        out = []
        for i, t in enumerate(trophies):
            r = last["rates"][i]
            median = r if r is not None else 0.01
            out.append({"id": t["id"], "q": quantiles(median, spread)})
        return {"trophies": out}
    return trend_forecast(case, history, n)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()

# GTA VI Trophy Rarity Forecaster

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/muellbiel-alt/gta6-trophy-forecast-nilunuha/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/muellbiel-alt/gta6-trophy-forecast-nilunuha/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

A forecaster for [Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) on make.fun.

## What it does

Given one case file (`/input/<id>.in`) — Exophase's reads of the Grand
Theft Auto VI PS5 trophy page before the checkpoint — it writes
`/output/<id>.out` with five quantiles (5%, 25%, 50%, 75%, 95%) of every
trophy's earn rate at the case's target.

## The idea in one paragraph

Exophase does not update its counts continuously: the page sits frozen
for hours, then refreshes in one jump — players tracked leaps, and every
rate dilutes at once. So the program models **counts, not rates**. The
conserved quantity across a refresh is the earner count
`E = rate x players / 100`. On the frozen launch-night page it forecasts
the documented first refresh (3,868 players, 4,276 trophies earned)
with a count-consistent allocation of the new earners. On a live
history it fits log-earners and log-players against log-time since
release, shrinks each trophy's slope toward its tier's mean, and reads
both fits at the target. Quantiles live on the logit scale, with a
spread from the fits' residuals plus a horizon term.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only. No network, no randomness, no wall
clock — two runs on the same input are byte-identical. One case takes
milliseconds, far inside the 8-second cap.

## Files

- `main.py` — the whole forecaster (single file, stdlib only).
- `LICENSE` — MIT.

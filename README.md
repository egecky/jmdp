# JMDP Experiment Code

This directory contains the code used for the finite-state and ALE experiments.

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

## Finite-State Experiments

```bash
bash scripts/run_tabular.sh
```

Outputs are written under `runs/tabular` by default.

## ALE Experiments

The ALE scripts train a fixed-policy moment model and then evaluate action-gap moments by Monte Carlo from cloned emulator states.

```bash
bash scripts/run_ale_4gpu.sh
```

The default launcher uses four GPUs and writes to `runs/ale`. The default ALE
configuration is the reward-coupled gap diagnostic used in the rebuttal:
`SIGMA_MODE=covgram`, `REWARD_BONUS=1.0`, `GAP_WEIGHT=0.1`,
`N_STATES=40`, `N_MC=32`, and `HORIZON=150`. Override settings with
environment variables, for example:

```bash
STEPS=75000 N_STATES=80 N_MC=128 HORIZON=200 GAP_WEIGHT=0.1 bash scripts/run_ale_4gpu.sh
```

To print the ALE summaries:

```bash
python scripts/summarize_ale.py --root runs/ale
```

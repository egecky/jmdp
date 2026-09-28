# Joint MDPs and Reinforcement Learning in Coupled-Dynamics Environments

Code for the paper:

> Ege C. Kaya, Mahsa Ghasemi, and Abolfazl Hashemi. “Joint MDPs and
> Reinforcement Learning in Coupled-Dynamics Environments.” UAI 2026.
> [arXiv:2603.06946](https://arxiv.org/abs/2603.06946)

## Overview

A standard MDP specifies the reward and transition law separately for each
action. It does not specify how the one-step outcomes of different actions are
coupled under shared randomness. That distinction matters for quantities such
as return gaps and probabilities that one action outperforms another, which
depend on a joint law rather than on the marginal return laws alone.

The paper introduces a joint MDP (JMDP), which augments an MDP with a
multi-action generative interface that samples counterfactual one-step outcomes
under shared exogenous randomness. The analysis uses a **local one-step
coupling**: outcomes at the queried state share randomness, while future
successor-state tables are sampled freshly and recursively. Under this model,
the paper derives Bellman operators for finite-order mixed return moments and
studies dynamic-programming and incremental algorithms for fixed-policy
evaluation.

This repository contains tabular demonstrations and neural experiments for
estimating joint moments. It studies policy evaluation, not policy improvement
or end-to-end agent training.

## Repository contents

- `experiments_v2/`: tabular coupled-reward-chain, Windy Gridworld, and
  action-gap Windy Gridworld experiments.
- `deep_learning/joint_eval/`: neural JIPE-2 experiments, including ALE-based
  moment estimation and Monte Carlo evaluation of action-gap quantities.
- `scripts/run_tabular.sh`: runs the three tabular experiments.
- `scripts/run_ale_4gpu.sh`: launches the ALE experiments on four GPUs.
- `scripts/summarize_ale.py`: summarizes ALE run outputs.

## Setup

For the tabular experiments, Python 3.10 or newer is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib
```

The complete optional dependency set, including PyTorch and ALE support, is
listed in `requirements.txt`. Install it with:

```bash
python -m pip install -r requirements.txt
```

## Quick check

Run the small test of the coupled second-moment operator:

```bash
python -m unittest discover -s tests
```

It checks that actions driven by the same one-step randomness have a different
cross-action return moment than the independent-coupling calculation.

For a small run of all three tabular experiments:

```bash
OUT=runs/smoke CHAIN_STATES=4 WIDTH=4 HEIGHT=3 ITERS=2 GAP_ITERS=2 \
  N_MC=4 HORIZON=8 bash scripts/run_tabular.sh
```

This writes residual plots and the gap-evaluation summary under `runs/smoke/`.

## Tabular experiments

Run all tabular experiments from the repository root:

```bash
bash scripts/run_tabular.sh
```

Outputs are written to `runs/tabular/` by default. The output directory and
experiment settings can be changed with environment variables such as `OUT`,
`ITERS`, `WIDTH`, `HEIGHT`, and `N_MC`; see `scripts/run_tabular.sh` for the
complete list.

## ALE experiments

The ALE launcher runs Pong, BattleZone, Boxing, and Atlantis concurrently,
using visible GPU IDs 0 through 3. It requires four available GPUs, compatible
ALE/Gymnasium Atari installations, and the corresponding ROM access configured
in the local environment.

```bash
bash scripts/run_ale_4gpu.sh
python scripts/summarize_ale.py --root runs/ale
```

The launcher exposes settings such as `STEPS`, `N_STATES`, `N_MC`, `HORIZON`,
`SIGMA_MODE`, and `GAP_WEIGHT`. For example:

```bash
STEPS=75000 N_STATES=80 N_MC=128 HORIZON=200 \
  bash scripts/run_ale_4gpu.sh
```

The neural/ALE experiments are substantially more resource-intensive than the
tabular examples.

## Citation

```bibtex
@inproceedings{kaya2026jointmdp,
  title     = {Joint MDPs and Reinforcement Learning in Coupled-Dynamics Environments},
  author    = {Kaya, Ege C. and Ghasemi, Mahsa and Hashemi, Abolfazl},
  booktitle = {Proceedings of the Conference on Uncertainty in Artificial Intelligence},
  year      = {2026}
}
```

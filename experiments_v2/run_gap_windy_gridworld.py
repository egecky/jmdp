import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from .envs import make_goal_directed_policy_grid, make_windy_gridworld
from .jipe2_tabular import MomentBundle2, value_iteration_T2
from .plotting import plot_residuals


def _set_paper_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _policy_action(pi_sd, s):
    return int(np.argmax(pi_sd[int(s)]))


def _rollout_pair_returns(kernel, pi_sd, s1, a1, s2, a2, gamma, horizon, rng):
    s_left = int(s1)
    s_right = int(s2)
    a_left = int(a1)
    a_right = int(a2)
    ret_left = 0.0
    ret_right = 0.0
    g = 1.0
    for _ in range(horizon):
        if s_left == s_right:
            p_u = kernel.probs_by_state[s_left]
            u = int(rng.choice(len(p_u), p=p_u))
            (r_left, sp_left) = kernel.outcome_fn(s_left, a_left, u)
            (r_right, sp_right) = kernel.outcome_fn(s_right, a_right, u)
        else:
            p_left = kernel.probs_by_state[s_left]
            p_right = kernel.probs_by_state[s_right]
            u_left = int(rng.choice(len(p_left), p=p_left))
            u_right = int(rng.choice(len(p_right), p=p_right))
            (r_left, sp_left) = kernel.outcome_fn(s_left, a_left, u_left)
            (r_right, sp_right) = kernel.outcome_fn(s_right, a_right, u_right)
        ret_left += g * float(r_left)
        ret_right += g * float(r_right)
        g *= gamma
        s_left = int(sp_left)
        s_right = int(sp_right)
        a_left = _policy_action(pi_sd, s_left)
        a_right = _policy_action(pi_sd, s_right)
    return (float(ret_left), float(ret_right))


def _rollout_pair_returns_independent(kernel, pi_sd, s1, a1, s2, a2, gamma, horizon, rng):
    s_left = int(s1)
    s_right = int(s2)
    a_left = int(a1)
    a_right = int(a2)
    ret_left = 0.0
    ret_right = 0.0
    g = 1.0
    for _ in range(horizon):
        p_left = kernel.probs_by_state[s_left]
        p_right = kernel.probs_by_state[s_right]
        u_left = int(rng.choice(len(p_left), p=p_left))
        u_right = int(rng.choice(len(p_right), p=p_right))
        (r_left, sp_left) = kernel.outcome_fn(s_left, a_left, u_left)
        (r_right, sp_right) = kernel.outcome_fn(s_right, a_right, u_right)
        ret_left += g * float(r_left)
        ret_right += g * float(r_right)
        g *= gamma
        s_left = int(sp_left)
        s_right = int(sp_right)
        a_left = _policy_action(pi_sd, s_left)
        a_right = _policy_action(pi_sd, s_right)
    return (float(ret_left), float(ret_right))


def _mc_gap_samples(kernel, pi_sd, s, a, b, gamma, horizon, n_mc, rng):
    gaps = np.zeros(n_mc, dtype=np.float64)
    for i in range(n_mc):
        (za, zb) = _rollout_pair_returns(
            kernel=kernel,
            pi_sd=pi_sd,
            s1=s,
            a1=a,
            s2=s,
            a2=b,
            gamma=gamma,
            horizon=horizon,
            rng=rng,
        )
        gaps[i] = za - zb
    return gaps


def _mc_gap_samples_independent(kernel, pi_sd, s, a, b, gamma, horizon, n_mc, rng):
    gaps = np.zeros(n_mc, dtype=np.float64)
    for i in range(n_mc):
        (za, zb) = _rollout_pair_returns_independent(
            kernel=kernel,
            pi_sd=pi_sd,
            s1=s,
            a1=a,
            s2=s,
            a2=b,
            gamma=gamma,
            horizon=horizon,
            rng=rng,
        )
        gaps[i] = za - zb
    return gaps


class GapRow:
    def __init__(
        self,
        s,
        a_star,
        b,
        pred_mean,
        pred_var,
        pred_var_independent,
        mc_mean,
        mc_var,
        mc_p_le0,
        mc_mean_independent,
        mc_var_independent,
        mc_p_le0_independent,
        cantelli_ub,
        cantelli_ub_independent,
    ):
        self.s = s
        self.a_star = a_star
        self.b = b
        self.pred_mean = pred_mean
        self.pred_var = pred_var
        self.pred_var_independent = pred_var_independent
        self.mc_mean = mc_mean
        self.mc_var = mc_var
        self.mc_p_le0 = mc_p_le0
        self.mc_mean_independent = mc_mean_independent
        self.mc_var_independent = mc_var_independent
        self.mc_p_le0_independent = mc_p_le0_independent
        self.cantelli_ub = cantelli_ub
        self.cantelli_ub_independent = cantelli_ub_independent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=10)
    ap.add_argument("--height", type=int, default=7)
    ap.add_argument("--p_wind", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--jipe_iters", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--outdir", type=str, default="results_v2")
    ap.add_argument("--n_states_eval", type=int, default=0, help="0 => all non-goal states")
    ap.add_argument("--n_mc", type=int, default=200)
    ap.add_argument("--mc_horizon", type=int, default=300)
    ap.add_argument("--alt_mode", type=str, choices=["all", "worst"], default="all")
    ap.add_argument("--progress_every", type=int, default=10)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (kernel, nS, nA, s_goal) = make_windy_gridworld(
        width=args.width, height=args.height, p_wind=args.p_wind
    )
    pi = make_goal_directed_policy_grid(args.width, args.height)
    (M, residuals) = value_iteration_T2(
        kernel=kernel,
        pi=pi,
        gamma=args.gamma,
        n_iters=args.jipe_iters,
        seed=args.seed,
        log_every=args.log_every,
        verbose=True,
    )
    plot_residuals(residuals, str(outdir / "windy_grid_residual.png"))
    all_states = [s for s in range(nS) if s != s_goal]
    rng = np.random.default_rng(args.seed)
    if args.n_states_eval and args.n_states_eval > 0:
        states = list(
            rng.choice(all_states, size=min(args.n_states_eval, len(all_states)), replace=False)
        )
    else:
        states = all_states
    rows = []
    for (idx, s) in enumerate(states):
        a_star = int(np.argmax(pi[s]))
        if args.alt_mode == "worst":
            mu_s = M.M_mu[s]
            b_list = [int(np.argmin(mu_s + 1000000000.0 * (np.arange(nA) == a_star)))]
        else:
            b_list = [b for b in range(nA) if b != a_star]
        for b in b_list:
            mu_a = float(M.M_mu[s, a_star])
            mu_b = float(M.M_mu[s, b])
            pred_mean = mu_a - mu_b
            S_aa = float(M.M_Sigma[s, a_star, s, a_star])
            S_bb = float(M.M_Sigma[s, b, s, b])
            S_ab = float(M.M_Sigma[s, a_star, s, b])
            pred_EG2 = S_aa + S_bb - 2.0 * S_ab
            pred_var = max(0.0, pred_EG2 - pred_mean * pred_mean)
            var_a = max(0.0, S_aa - mu_a * mu_a)
            var_b = max(0.0, S_bb - mu_b * mu_b)
            pred_var_independent = var_a + var_b
            gaps = _mc_gap_samples(
                kernel=kernel,
                pi_sd=pi,
                s=int(s),
                a=int(a_star),
                b=int(b),
                gamma=float(args.gamma),
                horizon=int(args.mc_horizon),
                n_mc=int(args.n_mc),
                rng=rng,
            )
            gaps_ind = _mc_gap_samples_independent(
                kernel=kernel,
                pi_sd=pi,
                s=int(s),
                a=int(a_star),
                b=int(b),
                gamma=float(args.gamma),
                horizon=int(args.mc_horizon),
                n_mc=int(args.n_mc),
                rng=rng,
            )
            mc_mean = float(np.mean(gaps))
            mc_var = float(np.var(gaps))
            mc_p_le0 = float(np.mean(gaps <= 0.0))
            mc_mean_ind = float(np.mean(gaps_ind))
            mc_var_ind = float(np.var(gaps_ind))
            mc_p_le0_ind = float(np.mean(gaps_ind <= 0.0))
            if pred_mean > 0.0 and pred_var >= 0.0:
                cantelli_ub = float(pred_var / (pred_var + pred_mean * pred_mean + 1e-12))
            else:
                cantelli_ub = 1.0
            if pred_mean > 0.0 and pred_var_independent >= 0.0:
                cantelli_ub_ind = float(
                    pred_var_independent / (pred_var_independent + pred_mean * pred_mean + 1e-12)
                )
            else:
                cantelli_ub_ind = 1.0
            rows.append(
                GapRow(
                    s=int(s),
                    a_star=int(a_star),
                    b=int(b),
                    pred_mean=float(pred_mean),
                    pred_var=float(pred_var),
                    pred_var_independent=float(pred_var_independent),
                    mc_mean=float(mc_mean),
                    mc_var=float(mc_var),
                    mc_p_le0=float(mc_p_le0),
                    mc_mean_independent=float(mc_mean_ind),
                    mc_var_independent=float(mc_var_ind),
                    mc_p_le0_independent=float(mc_p_le0_ind),
                    cantelli_ub=float(cantelli_ub),
                    cantelli_ub_independent=float(cantelli_ub_ind),
                )
            )
        if args.progress_every and args.progress_every > 0:
            if idx == 0 or (idx + 1) % args.progress_every == 0 or idx == len(states) - 1:
                print(f"gap-eval: processed {idx + 1}/{len(states)} states", flush=True)
    csv_path = outdir / "gap_eval_windy_gridworld.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "s",
                "a_star",
                "b",
                "pred_mean",
                "pred_var",
                "pred_var_independent",
                "mc_mean",
                "mc_var",
                "mc_p_le0",
                "mc_mean_independent",
                "mc_var_independent",
                "mc_p_le0_independent",
                "cantelli_ub",
                "cantelli_ub_independent",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.s,
                    r.a_star,
                    r.b,
                    r.pred_mean,
                    r.pred_var,
                    r.pred_var_independent,
                    r.mc_mean,
                    r.mc_var,
                    r.mc_p_le0,
                    r.mc_mean_independent,
                    r.mc_var_independent,
                    r.mc_p_le0_independent,
                    r.cantelli_ub,
                    r.cantelli_ub_independent,
                ]
            )
    pred_mean = np.array([r.pred_mean for r in rows])
    mc_mean = np.array([r.mc_mean for r in rows])
    pred_var = np.array([r.pred_var for r in rows])
    pred_var_ind = np.array([r.pred_var_independent for r in rows])
    mc_var = np.array([r.mc_var for r in rows])
    mc_var_ind = np.array([r.mc_var_independent for r in rows])
    mc_tail = np.array([r.mc_p_le0 for r in rows])
    cant_ub = np.array([r.cantelli_ub for r in rows])
    mc_tail_ind = np.array([r.mc_p_le0_independent for r in rows])
    cant_ub_ind = np.array([r.cantelli_ub_independent for r in rows])

    def _rmse(x, y):
        return float(np.sqrt(np.mean((np.asarray(x) - np.asarray(y)) ** 2)))

    def _corr(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    summary = {
        "n_rows": int(len(rows)),
        "mc_semantics": "recursive_pair_process",
        "mean_rmse": _rmse(pred_mean, mc_mean),
        "mean_corr": _corr(pred_mean, mc_mean),
        "var_rmse": _rmse(pred_var, mc_var),
        "var_corr": _corr(pred_var, mc_var),
        "independent_var_rmse_against_coupled_mc": _rmse(pred_var_ind, mc_var),
        "independent_var_corr_against_coupled_mc": _corr(pred_var_ind, mc_var),
        "independent_mc_var_rmse_against_coupled_mc": _rmse(mc_var_ind, mc_var),
        "tail_bound_violation_rate": float(np.mean(mc_tail > cant_ub + 1e-12)),
        "independent_tail_bound_violation_rate": float(np.mean(mc_tail_ind > cant_ub_ind + 1e-12)),
    }
    summary_path = outdir / "gap_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    def scatter(x, y, xlabel, ylabel, path):
        (fig, ax) = plt.subplots(figsize=(4.0, 3.5))
        ax.scatter(x, y, s=8)
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        ax.plot([lo, hi], [lo, hi], linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        _set_paper_axes(ax)
        fig.savefig(path)
        plt.close(fig)

    scatter(
        pred_mean, mc_mean, "Predicted gap mean", "MC gap mean", outdir / "gap_mean_scatter.png"
    )
    scatter(
        pred_var,
        mc_var,
        "Predicted gap variance",
        "MC gap variance",
        outdir / "gap_var_scatter.png",
    )
    scatter(
        pred_var_ind,
        mc_var,
        "Independent-marginals variance",
        "Coupled MC gap variance",
        outdir / "gap_var_independent_baseline.png",
    )
    scatter(cant_ub, mc_tail, "Cantelli upper bound", "MC P(G<=0)", outdir / "gap_tail_scatter.png")
    scatter(
        cant_ub_ind,
        mc_tail_ind,
        "Independent Cantelli upper bound",
        "Independent MC P(G<=0)",
        outdir / "gap_tail_independent_baseline.png",
    )
    print(f"wrote: {csv_path}")
    print(f"wrote: {summary_path}")


if __name__ == "__main__":
    main()

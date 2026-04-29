import argparse
import json
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
from .coupled_atari_env import CoupledAtariJMDP, CouplingConfig
from .networks import JIPE2Net


def sample_policy_action(rng, nA, policy):
    if policy == "random":
        return int(rng.integers(0, nA))
    raise ValueError(f"Unsupported policy: {policy}")


class StateRecord:
    def __init__(self, obs, snapshot, last_action):
        self.obs = obs
        self.snapshot = snapshot
        self.last_action = int(last_action)


def rollout_from_snapshot(env, snapshot, last_action, rng_seed, policy, gamma, ret_scale, horizon):
    env.restore_state(snapshot)
    env.last_executed_action = int(last_action)
    env.rng = np.random.default_rng(int(rng_seed))
    disc = 1.0
    total = 0.0
    for _ in range(int(horizon)):
        a = sample_policy_action(env.rng, env.nA, policy)
        (obs, r, done, _sid) = env.query_actions([a], execute_index=0)[0]
        total += disc * (float(r) / float(ret_scale))
        disc *= float(gamma)
        if done:
            break
    return float(total)


def rollout_pair_from_snapshots(
    env,
    snapshot_a,
    snapshot_b,
    last_action_a,
    last_action_b,
    state_id_a,
    state_id_b,
    rng_seed,
    policy,
    gamma,
    ret_scale,
    horizon,
    couple_matching_states=True,
    match_last_action=True,
):
    rng = np.random.default_rng(int(rng_seed))
    snap_a = snapshot_a
    snap_b = snapshot_b
    la = int(last_action_a)
    lb = int(last_action_b)
    sid_a = int(state_id_a)
    sid_b = int(state_id_b)
    active_a = snap_a is not None
    active_b = snap_b is not None
    disc = 1.0
    total_a = 0.0
    total_b = 0.0
    for _ in range(int(horizon)):
        if not active_a and (not active_b):
            break
        same_augmented_state = sid_a == sid_b and ((not match_last_action) or la == lb)
        shared = bool(couple_matching_states and active_a and active_b and same_augmented_state)
        if shared:
            env.restore_state(snap_a)
            env.last_executed_action = int(la)
            env.rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            act_a = sample_policy_action(env.rng, env.nA, policy)
            act_b = sample_policy_action(env.rng, env.nA, policy)
            outs = env.query_actions_with_states([act_a, act_b], execute_index=0)
            (_obs_a, r_a, done_a, sid_a_next, snap_a_next) = outs[0]
            (_obs_b, r_b, done_b, sid_b_next, snap_b_next) = outs[1]
            total_a += disc * (float(r_a) / float(ret_scale))
            total_b += disc * (float(r_b) / float(ret_scale))
            active_a = not bool(done_a)
            active_b = not bool(done_b)
            snap_a = None if done_a else snap_a_next
            snap_b = None if done_b else snap_b_next
            sid_a = int(sid_a_next)
            sid_b = int(sid_b_next)
            la = int(act_a)
            lb = int(act_b)
        else:
            if active_a:
                env.restore_state(snap_a)
                env.last_executed_action = int(la)
                env.rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
                act_a = sample_policy_action(env.rng, env.nA, policy)
                (_obs_a, r_a, done_a, sid_a_next, snap_a_next) = env.query_actions_with_states(
                    [act_a], execute_index=0
                )[0]
                total_a += disc * (float(r_a) / float(ret_scale))
                active_a = not bool(done_a)
                snap_a = None if done_a else snap_a_next
                sid_a = int(sid_a_next)
                la = int(act_a)
            if active_b:
                env.restore_state(snap_b)
                env.last_executed_action = int(lb)
                env.rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
                act_b = sample_policy_action(env.rng, env.nA, policy)
                (_obs_b, r_b, done_b, sid_b_next, snap_b_next) = env.query_actions_with_states(
                    [act_b], execute_index=0
                )[0]
                total_b += disc * (float(r_b) / float(ret_scale))
                active_b = not bool(done_b)
                snap_b = None if done_b else snap_b_next
                sid_b = int(sid_b_next)
                lb = int(act_b)
        disc *= float(gamma)
    return (float(total_a), float(total_b))


def choose_pair(mu_all, rng, mode):
    nA = int(mu_all.shape[-1])
    if mode == "top2":
        order = np.argsort(-mu_all)
        (a, b) = (int(order[0]), int(order[1]))
        return (a, b)
    if mode == "random":
        a = int(rng.integers(0, nA))
        b = int(rng.integers(0, nA - 1))
        if b >= a:
            b += 1
        return (a, b)
    raise ValueError(f"Unknown pair_mode: {mode}")


def cantelli_upper_bound(mu, var):
    if mu <= 0.0:
        return 1.0
    denom = var + mu * mu
    if denom <= 1e-12:
        return 0.0
    return float(var / denom)


def _scatter_plot(x, y, outpath, xlabel, ylabel, title=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.05 * (hi - lo + 1e-12)
    lo -= pad
    hi += pad
    fig = plt.figure(figsize=(3.0, 3.0), dpi=200)
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=6)
    ax.plot([lo, hi], [lo, hi], linewidth=0.75)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    if title:
        ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(outpath)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="ALE/Breakout-v5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--policy", type=str, default="random", choices=["random"])
    p.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Optional: if provided, evaluate predicted moments vs MC; if omitted, MC-only.",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sigma_mode", type=str, default="pair", choices=["pair", "gram", "covgram"])
    p.add_argument("--gram_dim", type=int, default=128)
    p.add_argument("--n_states", type=int, default=200)
    p.add_argument("--collect_steps", type=int, default=5000)
    p.add_argument("--pair_mode", type=str, default="top2", choices=["top2", "random"])
    p.add_argument("--n_mc", type=int, default=64)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--log_every_state", type=int, default=1)
    p.add_argument("--log_every_mc", type=int, default=0)
    p.add_argument("--outdir", type=str, default="results/gap_eval")
    p.add_argument(
        "--coupling_mode",
        type=str,
        default="both",
        choices=["none", "reward", "rewards", "transition", "transitions", "both"],
    )
    p.add_argument(
        "--mc_semantics",
        type=str,
        default="recursive_pair",
        choices=["recursive_pair", "one_step_only"],
        help="Monte Carlo semantics for the continuation rollout.",
    )
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    ret_scale = 1.0 / (1.0 - float(args.gamma))
    coupling_mode = {"rewards": "reward", "transitions": "transition"}.get(
        args.coupling_mode, args.coupling_mode
    )
    match_last_action = coupling_mode in ("transition", "both")
    cfg = CouplingConfig(mode=coupling_mode)
    env = CoupledAtariJMDP(env_id=args.env_id, seed=args.seed, config=cfg)
    rng = np.random.default_rng(args.seed)
    obs0 = env.reset()
    nA = env.nA
    model = None
    if args.model_path:
        print(f"[gap-eval] loading model from: {args.model_path}", flush=True)
        model = JIPE2Net(n_actions=nA, sigma_mode=args.sigma_mode, gram_dim=args.gram_dim)
        ckpt = torch.load(args.model_path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "model" in ckpt and isinstance(ckpt["model"], dict):
                sd = ckpt["model"]
            elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                sd = ckpt["state_dict"]
            elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
                sd = ckpt["model_state_dict"]
            else:
                sd = ckpt
        else:
            sd = ckpt
        if isinstance(sd, dict) and any((k.startswith("module.") for k in sd.keys())):
            sd = {k.replace("module.", "", 1): v for (k, v) in sd.items()}
        model.load_state_dict(sd)
        model.to(args.device)
        model.eval()
    else:
        print("[gap-eval] no --model_path provided; running MC-only gap evaluation.", flush=True)
    records: List[StateRecord] = []
    for t in range(int(args.collect_steps)):
        if len(records) < int(args.n_states):
            records.append(
                StateRecord(
                    obs=env.obs.copy(),
                    snapshot=env.clone_state(),
                    last_action=int(env.last_executed_action),
                )
            )
        a = sample_policy_action(rng, nA, args.policy)
        (_obs1, _r, done, _sid) = env.query_actions([a], execute_index=0)[0]
        if done:
            env.reset()
        if len(records) >= int(args.n_states) and t >= int(args.collect_steps) - 1:
            break
        if len(records) >= int(args.n_states) and t >= int(args.collect_steps) // 2:
            break
    if len(records) < int(args.n_states):
        print(
            f"[gap-eval] warning: collected only {len(records)} states (requested {args.n_states})"
        )
    rows = []
    for (idx, rec) in enumerate(records):
        if args.log_every_state and (idx == 0 or (idx + 1) % args.log_every_state == 0):
            print(f"[gap-eval] state {idx + 1}/{len(records)}", flush=True)
        a = b = None
        g_mean_pred = g_var_pred = p_cantelli = float("nan")
        mu_a = mu_b = sig_aa = sig_bb = sig_ab = float("nan")
        pred_var_a = pred_var_b = pred_cov_ab = pred_gap_second = float("nan")
        if model is not None:
            obs = torch.from_numpy(rec.obs).unsqueeze(0).to(args.device)
            with torch.no_grad():
                mu_all = model.mu_all(obs).squeeze(0).cpu().numpy()
            (a, b) = choose_pair(mu_all, rng=rng, mode=args.pair_mode)
            a_t = torch.tensor([a], device=args.device, dtype=torch.long)
            b_t = torch.tensor([b], device=args.device, dtype=torch.long)
            with torch.no_grad():
                mu_a = float(model.mu(obs, a_t).item())
                mu_b = float(model.mu(obs, b_t).item())
                sig_aa = float(model.Sigma(obs, a_t, obs, a_t).item())
                sig_bb = float(model.Sigma(obs, b_t, obs, b_t).item())
                sig_ab = float(model.Sigma(obs, a_t, obs, b_t).item())
            g_mean_pred = mu_a - mu_b
            g2_pred = sig_aa + sig_bb - 2.0 * sig_ab
            g_var_pred = max(0.0, g2_pred - g_mean_pred * g_mean_pred)
            pred_var_a = max(0.0, sig_aa - mu_a * mu_a)
            pred_var_b = max(0.0, sig_bb - mu_b * mu_b)
            pred_cov_ab = sig_ab - mu_a * mu_b
            pred_gap_second = g2_pred
            p_cantelli = cantelli_upper_bound(g_mean_pred, g_var_pred)
        else:
            a = int(rng.integers(0, nA))
            b = int(rng.integers(0, nA - 1))
            if b >= a:
                b += 1
        gaps = []
        gaps_ind = []
        za_vals = []
        zb_vals = []
        za_ind_vals = []
        zb_ind_vals = []
        for j in range(int(args.n_mc)):
            if args.log_every_mc and (j == 0 or (j + 1) % args.log_every_mc == 0):
                print(f"  [gap-eval]  mc {j + 1}/{int(args.n_mc)}", flush=True)
            env.rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            env.restore_state(rec.snapshot)
            env.last_executed_action = int(rec.last_action)
            outs = env.query_actions_with_states([a, b], execute_index=0)
            (obs_a, r_a, done_a, _sid_a, snap_a) = outs[0]
            (obs_b, r_b, done_b, _sid_b, snap_b) = outs[1]
            if args.mc_semantics == "recursive_pair":
                (cont_a, cont_b) = rollout_pair_from_snapshots(
                    env=env,
                    snapshot_a=None if done_a else snap_a,
                    snapshot_b=None if done_b else snap_b,
                    last_action_a=a,
                    last_action_b=b,
                    state_id_a=_sid_a,
                    state_id_b=_sid_b,
                    rng_seed=int(rng.integers(0, 2**31 - 1)),
                    policy=args.policy,
                    gamma=float(args.gamma),
                    ret_scale=float(ret_scale),
                    horizon=int(args.horizon),
                    couple_matching_states=True,
                    match_last_action=match_last_action,
                )
            else:
                seed_a = int(rng.integers(0, 2**31 - 1))
                seed_b = int(rng.integers(0, 2**31 - 1))
                cont_a = (
                    0.0
                    if done_a
                    else rollout_from_snapshot(
                        env=env,
                        snapshot=snap_a,
                        last_action=a,
                        rng_seed=seed_a,
                        policy=args.policy,
                        gamma=float(args.gamma),
                        ret_scale=float(ret_scale),
                        horizon=int(args.horizon),
                    )
                )
                cont_b = (
                    0.0
                    if done_b
                    else rollout_from_snapshot(
                        env=env,
                        snapshot=snap_b,
                        last_action=b,
                        rng_seed=seed_b,
                        policy=args.policy,
                        gamma=float(args.gamma),
                        ret_scale=float(ret_scale),
                        horizon=int(args.horizon),
                    )
                )
            z_a = float(r_a) / float(ret_scale) + float(args.gamma) * float(cont_a)
            z_b = float(r_b) / float(ret_scale) + float(args.gamma) * float(cont_b)
            gaps.append(z_a - z_b)
            za_vals.append(z_a)
            zb_vals.append(z_b)
            env.rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            env.restore_state(rec.snapshot)
            env.last_executed_action = int(rec.last_action)
            out_a_ind = env.query_actions_with_states([a, b], execute_index=0)[0]
            env.rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            env.restore_state(rec.snapshot)
            env.last_executed_action = int(rec.last_action)
            out_b_ind = env.query_actions_with_states([a, b], execute_index=1)[1]
            (_obs_ai, r_ai, done_ai, sid_ai, snap_ai) = out_a_ind
            (_obs_bi, r_bi, done_bi, sid_bi, snap_bi) = out_b_ind
            (cont_ai, cont_bi) = rollout_pair_from_snapshots(
                env=env,
                snapshot_a=None if done_ai else snap_ai,
                snapshot_b=None if done_bi else snap_bi,
                last_action_a=a,
                last_action_b=b,
                state_id_a=sid_ai,
                state_id_b=sid_bi,
                rng_seed=int(rng.integers(0, 2**31 - 1)),
                policy=args.policy,
                gamma=float(args.gamma),
                ret_scale=float(ret_scale),
                horizon=int(args.horizon),
                couple_matching_states=False,
            )
            z_ai = float(r_ai) / float(ret_scale) + float(args.gamma) * float(cont_ai)
            z_bi = float(r_bi) / float(ret_scale) + float(args.gamma) * float(cont_bi)
            gaps_ind.append(z_ai - z_bi)
            za_ind_vals.append(z_ai)
            zb_ind_vals.append(z_bi)
        gaps = np.asarray(gaps, dtype=np.float64)
        gaps_ind = np.asarray(gaps_ind, dtype=np.float64)
        za_vals = np.asarray(za_vals, dtype=np.float64)
        zb_vals = np.asarray(zb_vals, dtype=np.float64)
        za_ind_vals = np.asarray(za_ind_vals, dtype=np.float64)
        zb_ind_vals = np.asarray(zb_ind_vals, dtype=np.float64)
        g_mean_mc = float(gaps.mean())
        g_var_mc = float(gaps.var(ddof=0))
        p_le0_mc = float((gaps <= 0.0).mean())
        g_mean_ind = float(gaps_ind.mean())
        g_var_ind = float(gaps_ind.var(ddof=0))
        p_le0_ind = float((gaps_ind <= 0.0).mean())
        mc_mean_a = float(za_vals.mean())
        mc_mean_b = float(zb_vals.mean())
        mc_var_a = float(za_vals.var(ddof=0))
        mc_var_b = float(zb_vals.var(ddof=0))
        mc_cov_ab = float(np.mean((za_vals - mc_mean_a) * (zb_vals - mc_mean_b)))
        mc_mean_a_ind = float(za_ind_vals.mean())
        mc_mean_b_ind = float(zb_ind_vals.mean())
        mc_var_a_ind = float(za_ind_vals.var(ddof=0))
        mc_var_b_ind = float(zb_ind_vals.var(ddof=0))
        mc_cov_ind = float(np.mean((za_ind_vals - mc_mean_a_ind) * (zb_ind_vals - mc_mean_b_ind)))
        if model is not None:
            pred_var_ind = pred_var_a + pred_var_b
            p_cantelli_ind = cantelli_upper_bound(g_mean_pred, pred_var_ind)
        else:
            pred_var_ind = float("nan")
            p_cantelli_ind = float("nan")
        rows.append(
            (
                idx,
                a,
                b,
                g_mean_pred,
                g_mean_mc,
                g_var_pred,
                g_var_mc,
                p_cantelli,
                p_le0_mc,
                pred_var_ind,
                g_mean_ind,
                g_var_ind,
                p_cantelli_ind,
                p_le0_ind,
                mu_a,
                mu_b,
                sig_aa,
                sig_bb,
                sig_ab,
                pred_var_a,
                pred_var_b,
                pred_cov_ab,
                pred_gap_second,
                mc_mean_a,
                mc_mean_b,
                mc_var_a,
                mc_var_b,
                mc_cov_ab,
                mc_mean_a_ind,
                mc_mean_b_ind,
                mc_var_a_ind,
                mc_var_b_ind,
                mc_cov_ind,
            )
        )
        if (idx + 1) % 25 == 0:
            print(f"[gap-eval] {idx + 1:4d}/{len(records)} states processed")
    csv_path = os.path.join(args.outdir, "gap_eval.csv")
    header = [
        "idx",
        "a",
        "b",
        "pred_mean",
        "mc_mean",
        "pred_var",
        "mc_var",
        "cantelli_ub",
        "mc_p_le0",
        "pred_var_independent",
        "mc_mean_independent",
        "mc_var_independent",
        "cantelli_ub_independent",
        "mc_p_le0_independent",
        "pred_mu_a",
        "pred_mu_b",
        "pred_sig_aa",
        "pred_sig_bb",
        "pred_sig_ab",
        "pred_var_a",
        "pred_var_b",
        "pred_cov_ab",
        "pred_gap_second",
        "mc_mean_a",
        "mc_mean_b",
        "mc_var_a",
        "mc_var_b",
        "mc_cov_ab",
        "mc_mean_a_independent",
        "mc_mean_b_independent",
        "mc_var_a_independent",
        "mc_var_b_independent",
        "mc_cov_independent",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join((str(x) for x in r)) + "\n")
    print(f"[gap-eval] wrote {csv_path}")
    arr = np.asarray(rows, dtype=np.float64)
    summary = {
        "n_rows": int(len(rows)),
        "env_id": str(args.env_id),
        "mc_semantics": str(args.mc_semantics),
        "sigma_mode": str(args.sigma_mode),
        "pair_mode": str(args.pair_mode),
        "coupling_mode": str(coupling_mode),
        "n_mc": int(args.n_mc),
        "horizon": int(args.horizon),
        "model_path": str(args.model_path),
    }

    def _rmse(x, y):
        return float(np.sqrt(np.nanmean((np.asarray(x) - np.asarray(y)) ** 2)))

    def _corr(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 2 or np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
            return float("nan")
        return float(np.corrcoef(x[mask], y[mask])[0, 1])

    if model is None:
        summary.update(
            {
                "mc_independent_var_rmse_against_coupled_mc": _rmse(arr[:, 11], arr[:, 6]),
                "mc_independent_var_corr_against_coupled_mc": _corr(arr[:, 11], arr[:, 6]),
                "mc_gap_var_decomposition_rmse": _rmse(
                    arr[:, 25] + arr[:, 26] - 2.0 * arr[:, 27], arr[:, 6]
                ),
            }
        )
        summary_path = os.path.join(args.outdir, "gap_eval_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"[gap-eval] wrote {summary_path}")
        env.close()
        return
    (pred_mean, mc_mean) = (arr[:, 3], arr[:, 4])
    (pred_var, mc_var) = (arr[:, 5], arr[:, 6])
    (cant_ub, mc_p) = (arr[:, 7], arr[:, 8])
    (pred_var_ind, mc_var_ind) = (arr[:, 9], arr[:, 11])
    (cant_ub_ind, mc_p_ind) = (arr[:, 12], arr[:, 13])
    (pred_var_a, pred_var_b, pred_cov) = (arr[:, 19], arr[:, 20], arr[:, 21])
    (mc_var_a, mc_var_b, mc_cov) = (arr[:, 25], arr[:, 26], arr[:, 27])
    summary.update(
        {
            "mean_rmse": _rmse(pred_mean, mc_mean),
            "mean_corr": _corr(pred_mean, mc_mean),
            "var_rmse": _rmse(pred_var, mc_var),
            "var_corr": _corr(pred_var, mc_var),
            "independent_var_rmse_against_coupled_mc": _rmse(pred_var_ind, mc_var),
            "independent_var_corr_against_coupled_mc": _corr(pred_var_ind, mc_var),
            "independent_var_rmse_against_independent_mc": _rmse(pred_var_ind, mc_var_ind),
            "independent_var_corr_against_independent_mc": _corr(pred_var_ind, mc_var_ind),
            "mc_independent_var_rmse_against_coupled_mc": _rmse(mc_var_ind, mc_var),
            "diag_var_a_rmse": _rmse(pred_var_a, mc_var_a),
            "diag_var_b_rmse": _rmse(pred_var_b, mc_var_b),
            "diag_var_sum_rmse": _rmse(pred_var_a + pred_var_b, mc_var_a + mc_var_b),
            "cov_ab_rmse": _rmse(pred_cov, mc_cov),
            "cov_ab_corr": _corr(pred_cov, mc_cov),
            "mc_gap_var_decomposition_rmse": _rmse(mc_var_a + mc_var_b - 2.0 * mc_cov, mc_var),
            "tail_bound_violation_rate": float(np.mean(mc_p > cant_ub + 1e-12)),
            "independent_tail_bound_violation_rate": float(np.mean(mc_p_ind > cant_ub_ind + 1e-12)),
        }
    )
    summary_path = os.path.join(args.outdir, "gap_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[gap-eval] wrote {summary_path}")
    _scatter_plot(
        pred_mean,
        mc_mean,
        os.path.join(args.outdir, "gap_mean_scatter.pdf"),
        "Predicted gap mean",
        "Recursive-pair MC gap mean",
    )
    _scatter_plot(
        pred_var,
        mc_var,
        os.path.join(args.outdir, "gap_var_scatter.pdf"),
        "Predicted gap variance",
        "Recursive-pair MC gap variance",
    )
    _scatter_plot(
        cant_ub,
        mc_p,
        os.path.join(args.outdir, "gap_tail_scatter.pdf"),
        "Predicted Cantelli upper bound",
        "MC P(G<=0)",
    )
    _scatter_plot(
        pred_var_ind,
        mc_var,
        os.path.join(args.outdir, "gap_var_independent_baseline.pdf"),
        "Independent-marginals variance baseline",
        "Recursive-pair MC gap variance",
    )
    _scatter_plot(
        pred_var_ind,
        mc_var_ind,
        os.path.join(args.outdir, "gap_var_independent_vs_independent_mc.pdf"),
        "Independent-marginals variance baseline",
        "Independent MC gap variance",
    )
    _scatter_plot(
        pred_var_a + pred_var_b,
        mc_var_a + mc_var_b,
        os.path.join(args.outdir, "gap_diag_var_sum_diagnostic.pdf"),
        "Predicted branch variance sum",
        "MC branch variance sum",
    )
    _scatter_plot(
        pred_cov,
        mc_cov,
        os.path.join(args.outdir, "gap_cov_diagnostic.pdf"),
        "Predicted branch covariance",
        "MC branch covariance",
    )
    _scatter_plot(
        cant_ub_ind,
        mc_p_ind,
        os.path.join(args.outdir, "gap_tail_independent_baseline.pdf"),
        "Independent-baseline Cantelli bound",
        "Independent MC P(G<=0)",
    )
    _scatter_plot(
        pred_mean,
        mc_mean,
        os.path.join(args.outdir, "gap_mean_scatter.png"),
        "Predicted gap mean",
        "Recursive-pair MC gap mean",
    )
    _scatter_plot(
        pred_var,
        mc_var,
        os.path.join(args.outdir, "gap_var_scatter.png"),
        "Predicted gap variance",
        "Recursive-pair MC gap variance",
    )
    _scatter_plot(
        cant_ub,
        mc_p,
        os.path.join(args.outdir, "gap_tail_scatter.png"),
        "Predicted Cantelli upper bound",
        "MC P(G<=0)",
    )
    _scatter_plot(
        pred_var_ind,
        mc_var,
        os.path.join(args.outdir, "gap_var_independent_baseline.png"),
        "Independent-marginals variance baseline",
        "Recursive-pair MC gap variance",
    )
    _scatter_plot(
        pred_var_ind,
        mc_var_ind,
        os.path.join(args.outdir, "gap_var_independent_vs_independent_mc.png"),
        "Independent-marginals variance baseline",
        "Independent MC gap variance",
    )
    _scatter_plot(
        pred_var_a + pred_var_b,
        mc_var_a + mc_var_b,
        os.path.join(args.outdir, "gap_diag_var_sum_diagnostic.png"),
        "Predicted branch variance sum",
        "MC branch variance sum",
    )
    _scatter_plot(
        pred_cov,
        mc_cov,
        os.path.join(args.outdir, "gap_cov_diagnostic.png"),
        "Predicted branch covariance",
        "MC branch covariance",
    )
    _scatter_plot(
        cant_ub_ind,
        mc_p_ind,
        os.path.join(args.outdir, "gap_tail_independent_baseline.png"),
        "Independent-baseline Cantelli bound",
        "Independent MC P(G<=0)",
    )
    env.close()


if __name__ == "__main__":
    main()

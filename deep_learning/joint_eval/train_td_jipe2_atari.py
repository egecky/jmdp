import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from .coupled_atari_env import CoupledAtariJMDP, CouplingConfig
from .networks import JIPE2Net


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_policy_action(rng, nA, policy):
    if policy == "random":
        return int(rng.integers(0, nA))
    if policy == "sticky0":
        return 0
    raise ValueError(f"Unknown policy={policy}")


def obs_to_uint8(obs):
    return np.clip(obs * 255.0, 0.0, 255.0).astype(np.uint8)


def obs_from_uint8(arr):
    return arr.astype(np.float32) / 255.0


class RingReplay:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.data = []
        self.i = 0

    def __len__(self):
        return len(self.data)

    def add(self, item):
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.i] = item
        self.i = (self.i + 1) % self.capacity

    def sample(self, batch_size, rng):
        idx = rng.integers(0, len(self.data), size=batch_size)
        return [self.data[int(j)] for j in idx]


def _plot_paper(series, outpath, ylabel):
    import matplotlib.pyplot as plt

    (fig, ax) = plt.subplots()
    ax.plot(series, linewidth=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="ALE/Breakout-v5")
    p.add_argument(
        "--mode", type=str, default="both", choices=["none", "reward", "transition", "both"]
    )
    p.add_argument("--p_shared_sticky", type=float, default=0.25)
    p.add_argument("--reward_bonus", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--sigma_weight", type=float, default=1.0)
    p.add_argument("--sigma_cross_weight", type=float, default=1.0)
    p.add_argument("--sigma_diag_weight", type=float, default=1.0)
    p.add_argument("--gap_weight", type=float, default=0.0)
    p.add_argument("--sigma_mode", type=str, default="pair", choices=["pair", "gram", "covgram"])
    p.add_argument("--gram_dim", type=int, default=128)
    p.add_argument("--target_update", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--policy", type=str, default="random", choices=["random", "sticky0"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="results_v2")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--replay_size", type=int, default=50000)
    p.add_argument("--start_learning", type=int, default=5000)
    p.add_argument("--updates_per_step", type=int, default=1)
    p.add_argument("--normalize_returns", action="store_true", default=True)
    p.add_argument("--symmetry_penalty", type=float, default=0.0)
    args = p.parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = CouplingConfig(
        mode=args.mode, p_shared_sticky=args.p_shared_sticky, reward_bonus=args.reward_bonus
    )
    env = CoupledAtariJMDP(env_id=args.env_id, seed=args.seed, config=cfg)
    nA = env.nA
    model = JIPE2Net(n_actions=nA, sigma_mode=args.sigma_mode, gram_dim=args.gram_dim).to(
        args.device
    )
    model_tgt = JIPE2Net(n_actions=nA, sigma_mode=args.sigma_mode, gram_dim=args.gram_dim).to(
        args.device
    )
    model_tgt.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mu_buf = RingReplay(args.replay_size)
    pair_buf = RingReplay(args.replay_size)
    obs_np = env.reset()
    ret_scale = 1.0 / (1.0 - args.gamma)
    if not args.normalize_returns:
        ret_scale = 1.0
    td_mu_hist = []
    td_sig_same_hist = []
    td_sig_cross_hist = []
    td_sig_diag_hist = []
    td_sig_total_hist = []
    td_gap_hist = []
    for t in range(1, args.steps + 1):
        a = sample_policy_action(rng, nA, args.policy)
        b = int(rng.integers(0, nA - 1))
        if b >= a:
            b += 1
        outcomes = env.query_actions([a, b], execute_index=0)
        ((obs1_np, r1, done1, _sid1), (obs2_np, r2, done2, _sid2)) = outcomes
        obs_u8 = obs_to_uint8(obs_np)
        obs1_u8 = obs_to_uint8(obs1_np)
        obs2_u8 = obs_to_uint8(obs2_np)
        mu_buf.add((obs_u8, a, float(r1), obs1_u8, bool(done1)))
        pair_buf.add(
            (obs_u8, a, b, float(r1), float(r2), obs1_u8, obs2_u8, bool(done1), bool(done2))
        )
        obs_np = env.obs
        if done1:
            obs_np = env.reset()
        if len(mu_buf) >= args.start_learning and len(pair_buf) >= args.start_learning:
            for _ in range(args.updates_per_step):
                batch_mu = mu_buf.sample(args.batch_size, rng)
                batch_pair = pair_buf.sample(args.batch_size, rng)
                obs_b = torch.from_numpy(
                    np.stack([obs_from_uint8(x[0]) for x in batch_mu], axis=0)
                ).to(args.device)
                act_b = torch.tensor([x[1] for x in batch_mu], device=args.device, dtype=torch.long)
                r_b = (
                    torch.tensor([x[2] for x in batch_mu], device=args.device, dtype=torch.float32)
                    / ret_scale
                )
                next_obs_b = torch.from_numpy(
                    np.stack([obs_from_uint8(x[3]) for x in batch_mu], axis=0)
                ).to(args.device)
                done_b = torch.tensor(
                    [x[4] for x in batch_mu], device=args.device, dtype=torch.float32
                )
                m_b = 1.0 - done_b
                a_next = torch.tensor(
                    [sample_policy_action(rng, nA, args.policy) for _ in range(args.batch_size)],
                    device=args.device,
                    dtype=torch.long,
                )
                with torch.no_grad():
                    mu_next = model_tgt.mu(next_obs_b, a_next)
                    target_mu = r_b + args.gamma * m_b * mu_next
                pred_mu = model.mu(obs_b, act_b)
                loss_mu = F.mse_loss(pred_mu, target_mu)
                obs_p = torch.from_numpy(
                    np.stack([obs_from_uint8(x[0]) for x in batch_pair], axis=0)
                ).to(args.device)
                a_p = torch.tensor([x[1] for x in batch_pair], device=args.device, dtype=torch.long)
                b_p = torch.tensor([x[2] for x in batch_pair], device=args.device, dtype=torch.long)
                r1_p = (
                    torch.tensor(
                        [x[3] for x in batch_pair], device=args.device, dtype=torch.float32
                    )
                    / ret_scale
                )
                r2_p = (
                    torch.tensor(
                        [x[4] for x in batch_pair], device=args.device, dtype=torch.float32
                    )
                    / ret_scale
                )
                next1_p = torch.from_numpy(
                    np.stack([obs_from_uint8(x[5]) for x in batch_pair], axis=0)
                ).to(args.device)
                next2_p = torch.from_numpy(
                    np.stack([obs_from_uint8(x[6]) for x in batch_pair], axis=0)
                ).to(args.device)
                d1_p = torch.tensor(
                    [x[7] for x in batch_pair], device=args.device, dtype=torch.float32
                )
                d2_p = torch.tensor(
                    [x[8] for x in batch_pair], device=args.device, dtype=torch.float32
                )
                m1_p = 1.0 - d1_p
                m2_p = 1.0 - d2_p
                a1p = torch.tensor(
                    [sample_policy_action(rng, nA, args.policy) for _ in range(args.batch_size)],
                    device=args.device,
                    dtype=torch.long,
                )
                a2p = torch.tensor(
                    [sample_policy_action(rng, nA, args.policy) for _ in range(args.batch_size)],
                    device=args.device,
                    dtype=torch.long,
                )
                with torch.no_grad():
                    mu1_cur = model_tgt.mu(obs_p, a_p)
                    mu2_cur = model_tgt.mu(obs_p, b_p)
                    mu1_next = model_tgt.mu(next1_p, a1p)
                    mu2_next = model_tgt.mu(next2_p, a2p)
                    sig_next = model_tgt.Sigma(next1_p, a1p, next2_p, a2p)
                    sig11_next = model_tgt.Sigma(next1_p, a1p, next1_p, a1p)
                    sig22_next = model_tgt.Sigma(next2_p, a2p, next2_p, a2p)
                    target_sig_same = (
                        r1_p * r2_p
                        + args.gamma * r1_p * m2_p * mu2_next
                        + args.gamma * r2_p * m1_p * mu1_next
                        + args.gamma**2 * m1_p * m2_p * sig_next
                    )
                    target_cov_same = target_sig_same - mu1_cur * mu2_cur
                    dr_p = r1_p - r2_p
                    target_gap_second = (
                        dr_p * dr_p
                        + 2.0 * args.gamma * dr_p * (m1_p * mu1_next - m2_p * mu2_next)
                        + args.gamma**2
                        * (
                            m1_p * sig11_next
                            + m2_p * sig22_next
                            - 2.0 * m1_p * m2_p * sig_next
                        )
                    )
                pred_sig_same = (
                    model.covariance(obs_p, a_p, obs_p, b_p)
                    if args.sigma_mode == "covgram"
                    else model.Sigma(obs_p, a_p, obs_p, b_p)
                )
                target_same = target_cov_same if args.sigma_mode == "covgram" else target_sig_same
                loss_sig_same = F.mse_loss(pred_sig_same, target_same)
                pred_gap_second = (
                    model.Sigma(obs_p, a_p, obs_p, a_p)
                    + model.Sigma(obs_p, b_p, obs_p, b_p)
                    - 2.0 * model.Sigma(obs_p, a_p, obs_p, b_p)
                )
                loss_gap = F.mse_loss(pred_gap_second, target_gap_second)
                batch_mu2 = mu_buf.sample(args.batch_size, rng)
                obs1_c = torch.from_numpy(
                    np.stack([obs_from_uint8(x[0]) for x in batch_mu], axis=0)
                ).to(args.device)
                a1_c = torch.tensor([x[1] for x in batch_mu], device=args.device, dtype=torch.long)
                r1_c = (
                    torch.tensor([x[2] for x in batch_mu], device=args.device, dtype=torch.float32)
                    / ret_scale
                )
                next1_c = torch.from_numpy(
                    np.stack([obs_from_uint8(x[3]) for x in batch_mu], axis=0)
                ).to(args.device)
                d1_c = torch.tensor(
                    [x[4] for x in batch_mu], device=args.device, dtype=torch.float32
                )
                m1_c = 1.0 - d1_c
                obs2_c = torch.from_numpy(
                    np.stack([obs_from_uint8(x[0]) for x in batch_mu2], axis=0)
                ).to(args.device)
                a2_c = torch.tensor([x[1] for x in batch_mu2], device=args.device, dtype=torch.long)
                r2_c = (
                    torch.tensor([x[2] for x in batch_mu2], device=args.device, dtype=torch.float32)
                    / ret_scale
                )
                next2_c = torch.from_numpy(
                    np.stack([obs_from_uint8(x[3]) for x in batch_mu2], axis=0)
                ).to(args.device)
                d2_c = torch.tensor(
                    [x[4] for x in batch_mu2], device=args.device, dtype=torch.float32
                )
                m2_c = 1.0 - d2_c
                a1c = torch.tensor(
                    [sample_policy_action(rng, nA, args.policy) for _ in range(args.batch_size)],
                    device=args.device,
                    dtype=torch.long,
                )
                a2c = torch.tensor(
                    [sample_policy_action(rng, nA, args.policy) for _ in range(args.batch_size)],
                    device=args.device,
                    dtype=torch.long,
                )
                with torch.no_grad():
                    mu1c_cur = model_tgt.mu(obs1_c, a1_c)
                    mu2c_cur = model_tgt.mu(obs2_c, a2_c)
                    mu1c_next = model_tgt.mu(next1_c, a1c)
                    mu2c_next = model_tgt.mu(next2_c, a2c)
                    sigc_next = model_tgt.Sigma(next1_c, a1c, next2_c, a2c)
                    target_sig_cross = (
                        r1_c * r2_c
                        + args.gamma * r1_c * m2_c * mu2c_next
                        + args.gamma * r2_c * m1_c * mu1c_next
                        + args.gamma**2 * m1_c * m2_c * sigc_next
                    )
                    target_cov_cross = target_sig_cross - mu1c_cur * mu2c_cur
                pred_sig_cross = (
                    model.covariance(obs1_c, a1_c, obs2_c, a2_c)
                    if args.sigma_mode == "covgram"
                    else model.Sigma(obs1_c, a1_c, obs2_c, a2_c)
                )
                target_cross = (
                    target_cov_cross if args.sigma_mode == "covgram" else target_sig_cross
                )
                loss_sig_cross = F.mse_loss(pred_sig_cross, target_cross)
                (obs_d1, a_d1, r_d1, next_d1, m_d1) = (obs1_c, a1_c, r1_c, next1_c, m1_c)
                (obs_d2, a_d2, r_d2, next_d2, m_d2) = (obs1_c, a1_c, r1_c, next1_c, m1_c)
                adp = torch.tensor(
                    [sample_policy_action(rng, nA, args.policy) for _ in range(args.batch_size)],
                    device=args.device,
                    dtype=torch.long,
                )
                with torch.no_grad():
                    mu_d_cur = model_tgt.mu(obs_d1, a_d1)
                    mu_d = model_tgt.mu(next_d1, adp)
                    sig_d = model_tgt.Sigma(next_d1, adp, next_d1, adp)
                    target_sig_diag = (
                        r_d1 * r_d1
                        + 2.0 * args.gamma * r_d1 * m_d1 * mu_d
                        + args.gamma**2 * m_d1 * sig_d
                    )
                    target_cov_diag = target_sig_diag - mu_d_cur * mu_d_cur
                pred_sig_diag = (
                    model.covariance(obs_d1, a_d1, obs_d2, a_d2)
                    if args.sigma_mode == "covgram"
                    else model.Sigma(obs_d1, a_d1, obs_d2, a_d2)
                )
                target_diag = target_cov_diag if args.sigma_mode == "covgram" else target_sig_diag
                loss_sig_diag = F.mse_loss(pred_sig_diag, target_diag)
                sym_pen = torch.tensor(0.0, device=args.device)
                if args.symmetry_penalty > 0.0:
                    pred_swap = (
                        model.covariance(obs2_c, a2_c, obs1_c, a1_c)
                        if args.sigma_mode == "covgram"
                        else model.Sigma(obs2_c, a2_c, obs1_c, a1_c)
                    )
                    sym_pen = F.mse_loss(pred_sig_cross, pred_swap)
                loss = (
                    loss_mu
                    + args.sigma_weight * loss_sig_same
                    + args.sigma_cross_weight * loss_sig_cross
                    + args.sigma_diag_weight * loss_sig_diag
                    + args.gap_weight * loss_gap
                    + args.symmetry_penalty * sym_pen
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=10.0)
                opt.step()
                td_mu_hist.append(float(loss_mu.detach().cpu().item()))
                td_sig_same_hist.append(float(loss_sig_same.detach().cpu().item()))
                td_sig_cross_hist.append(float(loss_sig_cross.detach().cpu().item()))
                td_sig_diag_hist.append(float(loss_sig_diag.detach().cpu().item()))
                td_gap_hist.append(float(loss_gap.detach().cpu().item()))
                td_sig_total_hist.append(
                    float(
                        (
                            loss_sig_same
                            + loss_sig_cross
                            + loss_sig_diag
                            + args.gap_weight * loss_gap
                        )
                        .detach()
                        .cpu()
                        .item()
                    )
                )
        if t % args.target_update == 0:
            model_tgt.load_state_dict(model.state_dict())
        if t % args.log_every == 0 and len(td_mu_hist) >= 1:
            mu_ma = float(np.mean(td_mu_hist[-min(len(td_mu_hist), args.log_every) :]))
            sg1_ma = float(np.mean(td_sig_same_hist[-min(len(td_sig_same_hist), args.log_every) :]))
            sg2_ma = float(
                np.mean(td_sig_cross_hist[-min(len(td_sig_cross_hist), args.log_every) :])
            )
            sgd_ma = float(np.mean(td_sig_diag_hist[-min(len(td_sig_diag_hist), args.log_every) :]))
            sgt_ma = float(
                np.mean(td_sig_total_hist[-min(len(td_sig_total_hist), args.log_every) :])
            )
            gap_ma = float(np.mean(td_gap_hist[-min(len(td_gap_hist), args.log_every) :]))
            msg = (
                f"step {t:>7d} | TD(mu)={mu_ma:.4e} | TD(Sigma diag)={sgd_ma:.4e} "
                f"| TD(Sigma same)={sg1_ma:.4e} | TD(Sigma cross)={sg2_ma:.4e} "
                f"| TD(Sigma total)={sgt_ma:.4e} | TD(gap)={gap_ma:.4e} | buf={len(mu_buf)}"
            )
            print(msg, flush=True)
    outdir = os.path.join(args.outdir, "atari_jipe2")
    os.makedirs(outdir, exist_ok=True)
    metrics = {
        "args": vars(args),
        "coupling": vars(cfg),
        "td_mu": td_mu_hist,
        "td_sigma_same": td_sig_same_hist,
        "td_sigma_cross": td_sig_cross_hist,
        "td_sigma_diag": td_sig_diag_hist,
        "td_gap": td_gap_hist,
        "td_sigma_total": td_sig_total_hist,
        "ret_scale": ret_scale,
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    if len(td_mu_hist) > 1000:
        mu_smooth = np.convolve(td_mu_hist, np.ones(1000) / 1000, mode="valid")
        _plot_paper(mu_smooth, os.path.join(outdir, "td_mu.png"), "TD MSE (mean)")
    if len(td_sig_same_hist) > 1000:
        sg_smooth = np.convolve(td_sig_same_hist, np.ones(1000) / 1000, mode="valid")
        _plot_paper(
            sg_smooth, os.path.join(outdir, "td_sigma_same.png"), "TD MSE (Sigma, same-state)"
        )
    if len(td_sig_cross_hist) > 1000:
        sg_smooth = np.convolve(td_sig_cross_hist, np.ones(1000) / 1000, mode="valid")
        _plot_paper(
            sg_smooth, os.path.join(outdir, "td_sigma_cross.png"), "TD MSE (Sigma, cross-state)"
        )
    if len(td_sig_diag_hist) > 1000:
        sg_smooth = np.convolve(td_sig_diag_hist, np.ones(1000) / 1000, mode="valid")
        _plot_paper(
            sg_smooth, os.path.join(outdir, "td_sigma_diag.png"), "TD MSE (Sigma, diagonal)"
        )
    if len(td_sig_total_hist) > 1000:
        sg_smooth = np.convolve(td_sig_total_hist, np.ones(1000) / 1000, mode="valid")
        _plot_paper(sg_smooth, os.path.join(outdir, "td_sigma_total.png"), "TD MSE (Sigma, total)")
    torch.save({"model": model.state_dict()}, os.path.join(outdir, "weights.pt"))
    print(f"Saved outputs to: {outdir}", flush=True)


if __name__ == "__main__":
    main()

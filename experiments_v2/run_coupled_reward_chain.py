import argparse
import os
from .envs import make_coupled_reward_chain, make_uniform_policy
from .jipe2_tabular import value_iteration_T2
from .plotting import plot_residuals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nS", type=int, default=25)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--outdir", type=str, default="results_v2")
    args = p.parse_args()
    (kernel, nS, nA, terminal) = make_coupled_reward_chain(nS=args.nS)
    pi = make_uniform_policy(nA)
    (_, residuals) = value_iteration_T2(
        kernel,
        pi,
        gamma=args.gamma,
        n_iters=args.iters,
        seed=args.seed,
        log_every=args.log_every,
        verbose=True,
    )
    outpath = os.path.join(
        args.outdir, f"coupled_reward_chain_residual_nS{args.nS}_g{args.gamma}.png"
    )
    plot_residuals(residuals, outpath)
    print(f"Saved: {outpath}")
    print(f"Final residual: {float(residuals[-1]):.4e}")


if __name__ == "__main__":
    main()

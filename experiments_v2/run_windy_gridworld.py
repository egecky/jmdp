import argparse
import os
from .envs import make_windy_gridworld, make_uniform_policy
from .jipe2_tabular import value_iteration_T2
from .plotting import plot_residuals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=10)
    p.add_argument("--height", type=int, default=7)
    p.add_argument("--p_wind", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--outdir", type=str, default="results_v2")
    args = p.parse_args()
    (kernel, nS, nA, goal) = make_windy_gridworld(
        width=args.width, height=args.height, p_wind=args.p_wind
    )
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
        args.outdir,
        f"windy_gridworld_residual_{args.width}x{args.height}_p{args.p_wind}_g{args.gamma}.png",
    )
    plot_residuals(residuals, outpath)
    print(f"Saved: {outpath}")
    print(f"Final residual: {float(residuals[-1]):.4e}")


if __name__ == "__main__":
    main()

from typing import Callable, Tuple
import numpy as np
from .envs import TabularJointKernel


class MomentBundle2:
    def __init__(self, M_mu, M_Sigma):
        self.M_mu = M_mu
        self.M_Sigma = M_Sigma


def zero_moments(nS, nA):
    return MomentBundle2(
        M_mu=np.zeros((nS, nA), dtype=np.float64),
        M_Sigma=np.zeros((nS, nA, nS, nA), dtype=np.float64),
    )


def _is_state_dependent_policy(pi):
    return pi.ndim == 2


def V_from_mu(mu, pi):
    if _is_state_dependent_policy(pi):
        return np.sum(mu * pi, axis=1)
    return mu @ pi


def apply_T2(kernel, pi, gamma, M, one_step_coupling=True):
    (nS, nA) = (kernel.nS, kernel.nA)
    assert pi.shape == (nA,) or pi.shape == (nS, nA)
    mu = M.M_mu
    Sigma = M.M_Sigma
    Vmu = V_from_mu(mu, pi)
    new_mu = np.zeros_like(mu)
    for s in range(nS):
        p_u = kernel.probs_by_state[s]
        for a in range(nA):
            acc = 0.0
            for (u, pu) in enumerate(p_u):
                (r, sp) = kernel.outcome_fn(s, a, u)
                acc += pu * (r + gamma * Vmu[sp])
            new_mu[s, a] = acc
    new_Sigma = np.zeros_like(Sigma)

    def cont_expectation(sp1, sp2):
        if _is_state_dependent_policy(pi):
            pi1 = pi[sp1]
            pi2 = pi[sp2]
        else:
            pi1 = pi
            pi2 = pi
        return float(np.sum(Sigma[sp1, :, sp2, :] * (pi1[:, None] * pi2[None, :])))

    for s1 in range(nS):
        p1 = kernel.probs_by_state[s1]
        for a1 in range(nA):
            for s2 in range(nS):
                p2 = kernel.probs_by_state[s2]
                for a2 in range(nA):
                    acc = 0.0
                    if one_step_coupling and s1 == s2:
                        for (u, pu) in enumerate(p1):
                            (r1, sp1) = kernel.outcome_fn(s1, a1, u)
                            (r2, sp2) = kernel.outcome_fn(s2, a2, u)
                            acc += pu * (
                                r1 * r2
                                + gamma * r1 * Vmu[sp2]
                                + gamma * r2 * Vmu[sp1]
                                + gamma**2 * cont_expectation(sp1, sp2)
                            )
                    else:
                        for (u1, pu1) in enumerate(p1):
                            (r1, sp1) = kernel.outcome_fn(s1, a1, u1)
                            for (u2, pu2) in enumerate(p2):
                                (r2, sp2) = kernel.outcome_fn(s2, a2, u2)
                                pu = pu1 * pu2
                                acc += pu * (
                                    r1 * r2
                                    + gamma * r1 * Vmu[sp2]
                                    + gamma * r2 * Vmu[sp1]
                                    + gamma**2 * cont_expectation(sp1, sp2)
                                )
                    new_Sigma[s1, a1, s2, a2] = acc
    return MomentBundle2(M_mu=new_mu, M_Sigma=new_Sigma)


def residual_T2(kernel, pi, gamma, M, lam):
    TM = apply_T2(kernel, pi, gamma, M)
    d_mu = np.max(np.abs(TM.M_mu - M.M_mu))
    d_S = np.max(np.abs(TM.M_Sigma - M.M_Sigma))
    return float(max(d_mu, d_S / lam))


def value_iteration_T2(
    kernel, pi, gamma, n_iters=200, lam=None, seed=0, log_every=25, verbose=True
):
    rng = np.random.default_rng(seed)
    (nS, nA) = (kernel.nS, kernel.nA)
    M = MomentBundle2(
        M_mu=rng.normal(scale=0.1, size=(nS, nA)),
        M_Sigma=rng.normal(scale=0.1, size=(nS, nA, nS, nA)),
    )
    if lam is None:
        lam = 2.0 / (1.0 - gamma)
    residuals = np.zeros(n_iters, dtype=np.float64)
    for k in range(n_iters):
        res = residual_T2(kernel, pi, gamma, M, lam)
        residuals[k] = res
        if verbose and log_every and (log_every > 0):
            if k == 0 or (k + 1) % log_every == 0 or k == n_iters - 1:
                print(f"iter {k + 1:>5d}/{n_iters} | residual={res:.4e}")
        M = apply_T2(kernel, pi, gamma, M)
    return (M, residuals)

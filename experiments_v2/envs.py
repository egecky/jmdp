from typing import Callable, Dict, List, Sequence, Tuple
import numpy as np


class TabularJointKernel:
    def __init__(self, nS, nA, probs_by_state, outcome_fn):
        self.nS = int(nS)
        self.nA = int(nA)
        self.probs_by_state = probs_by_state
        self.outcome_fn = outcome_fn

    def nU(self, s):
        return int(self.probs_by_state[s].shape[0])

    def outcomes(self, s, u):
        return [self.outcome_fn(s, a, u) for a in range(self.nA)]


def make_coupled_reward_chain(nS=25, terminal_reward=1.0):
    nA = 2
    probs_by_state = [np.array([0.5, 0.5], dtype=np.float64) for _ in range(nS)]

    def outcome_fn(s, a, u):
        if s == nS - 1:
            return (float(terminal_reward), s)
        sp = min(s + 1, nS - 1)
        if a == 0:
            r = float(u)
        else:
            r = float(1 - u)
        return (r, sp)

    return (
        TabularJointKernel(nS=nS, nA=nA, probs_by_state=probs_by_state, outcome_fn=outcome_fn),
        nS,
        nA,
        nS - 1,
    )


def make_windy_gridworld(width=10, height=7, p_wind=0.2):
    nA = 4
    nS = width * height
    goal_xy = (width - 1, 0)
    probs_by_state = [np.array([1.0 - p_wind, p_wind], dtype=np.float64) for _ in range(nS)]

    def to_xy(s):
        return (s % width, s // width)

    def to_s(x, y):
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        return y * width + x

    def step_xy(x, y, a):
        if a == 0:
            return (x, y - 1)
        if a == 1:
            return (x + 1, y)
        if a == 2:
            return (x, y + 1)
        if a == 3:
            return (x - 1, y)
        raise ValueError("invalid action")

    def outcome_fn(s, a, u):
        if to_xy(s) == goal_xy:
            return (1.0, s)
        (x, y) = to_xy(s)
        (x2, y2) = step_xy(x, y, a)
        if u == 1:
            y2 = y2 + 1
        sp = to_s(x2, y2)
        r = 1.0 if to_xy(sp) == goal_xy else 0.0
        return (r, sp)

    s_goal = to_s(*goal_xy)
    return (
        TabularJointKernel(nS=nS, nA=nA, probs_by_state=probs_by_state, outcome_fn=outcome_fn),
        nS,
        nA,
        s_goal,
    )


def make_uniform_policy(nA):
    pi = np.ones(nA, dtype=np.float64) / float(nA)
    return pi


def make_eps_greedy_policy(q, eps=0.1):
    nA = q.shape[0]
    pi = np.ones(nA, dtype=np.float64) * (eps / nA)
    pi[int(np.argmax(q))] += 1.0 - eps
    return pi


def make_goal_directed_policy_grid(width, height):
    nA = 4
    nS = width * height
    (goal_x, goal_y) = (width - 1, 0)

    def to_xy(s):
        return (s % width, s // width)

    pi = np.zeros((nS, nA), dtype=np.float64)
    for s in range(nS):
        (x, y) = to_xy(s)
        if (x, y) == (goal_x, goal_y):
            a = 0
        elif x < goal_x:
            a = 1
        elif y > goal_y:
            a = 0
        else:
            a = 0
        pi[s, a] = 1.0
    return pi

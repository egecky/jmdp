from typing import List, Tuple
import numpy as np
import hashlib

try:
    import gymnasium as gym
except Exception:
    gym = None
try:
    import ale_py
    import gymnasium.envs.atari
except Exception:
    pass
from PIL import Image


def _preprocess_obs(obs_rgb, out_size=84):
    img = Image.fromarray(obs_rgb)
    img = img.convert("L")
    img = img.resize((out_size, out_size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[None, :, :]


def _normalize_reward(r):
    return float(np.clip(r, 0.0, 1.0))


class CouplingConfig:
    def __init__(self, mode="both", p_shared_sticky=0.25, reward_bonus=0.25):
        self.mode = mode
        self.p_shared_sticky = float(p_shared_sticky)
        self.reward_bonus = float(reward_bonus)


class CoupledAtariJMDP:
    def __init__(
        self,
        env_id="ALE/Breakout-v5",
        seed=0,
        config=None,
        frameskip=1,
        repeat_action_probability=0.0,
    ):
        if gym is None:
            raise RuntimeError("gymnasium is required for Atari experiments")
        self.env = gym.make(
            env_id,
            obs_type="rgb",
            frameskip=frameskip,
            repeat_action_probability=repeat_action_probability,
        )
        self.env.reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.config = config or CouplingConfig()
        self.nA = int(self.env.action_space.n)
        self.last_executed_action = 0
        self.obs = None

    def reset(self):
        (obs, _) = self.env.reset()
        self.obs = _preprocess_obs(obs)
        self.last_executed_action = 0
        return self.obs

    def _apply_transition_coupling(self, a, u_trans):
        if self.config.mode not in ("transition", "both"):
            return a
        if u_trans == 1:
            return int(self.last_executed_action)
        return a

    def _apply_reward_coupling(self, idx, u_rew, base_reward):
        r = float(base_reward)
        if self.config.mode in ("reward", "both"):
            if idx == 0:
                r = r + self.config.reward_bonus * float(u_rew)
            elif idx == 1:
                r = r - self.config.reward_bonus * float(u_rew)
        return _normalize_reward(r)

    def query_actions(self, actions, execute_index=0):
        if len(actions) < 1:
            raise ValueError("actions must be non-empty")
        if execute_index < 0 or execute_index >= len(actions):
            raise ValueError("invalid execute_index")
        u_trans = 1 if self.rng.random() < self.config.p_shared_sticky else 0
        u_rew = 1 if self.rng.random() < 0.5 else -1
        ale = self.env.unwrapped.ale
        snapshot = ale.cloneState()
        outcomes: List[Tuple[np.ndarray, float, bool, int]] = []
        snapshot_next = None
        for (idx, a) in enumerate(actions):
            a_eff = self._apply_transition_coupling(int(a), u_trans)
            (obs_rgb, r, terminated, truncated, _) = self.env.step(a_eff)
            done = bool(terminated or truncated)
            obs = _preprocess_obs(obs_rgb)
            r_norm = self._apply_reward_coupling(idx=idx, u_rew=u_rew, base_reward=float(r))
            ram = np.asarray(ale.getRAM(), dtype=np.uint8).tobytes()
            state_id = int.from_bytes(hashlib.blake2b(ram, digest_size=8).digest(), "little")
            outcomes.append((obs, r_norm, done, state_id))
            if idx == execute_index:
                snapshot_next = ale.cloneState()
            ale.restoreState(snapshot)
        assert snapshot_next is not None
        ale.restoreState(snapshot_next)
        self.obs = outcomes[execute_index][0]
        self.last_executed_action = int(actions[execute_index])
        return outcomes

    def clone_state(self):
        return self.env.unwrapped.ale.cloneState()

    def restore_state(self, snapshot):
        ale = self.env.unwrapped.ale
        ale.restoreState(snapshot)
        obs_rgb = ale.getScreenRGB()
        self.obs = _preprocess_obs(obs_rgb)

    def query_actions_with_states(self, actions, execute_index=0):
        if len(actions) < 1:
            raise ValueError("actions must be non-empty")
        if execute_index < 0 or execute_index >= len(actions):
            raise ValueError("invalid execute_index")
        u_trans = 1 if self.rng.random() < self.config.p_shared_sticky else 0
        u_rew = 1 if self.rng.random() < 0.5 else -1
        ale = self.env.unwrapped.ale
        snapshot0 = ale.cloneState()
        outcomes: List[Tuple[np.ndarray, float, bool, int, object]] = []
        snapshot_next = None
        for (idx, a) in enumerate(actions):
            a_eff = self._apply_transition_coupling(int(a), u_trans)
            (obs_rgb, r, terminated, truncated, _) = self.env.step(a_eff)
            done = bool(terminated or truncated)
            obs = _preprocess_obs(obs_rgb)
            r_norm = self._apply_reward_coupling(idx=idx, u_rew=u_rew, base_reward=float(r))
            ram = np.asarray(ale.getRAM(), dtype=np.uint8).tobytes()
            state_id = int.from_bytes(hashlib.blake2b(ram, digest_size=8).digest(), "little")
            succ = ale.cloneState()
            outcomes.append((obs, r_norm, done, state_id, succ))
            if idx == execute_index:
                snapshot_next = succ
            ale.restoreState(snapshot0)
        assert snapshot_next is not None
        ale.restoreState(snapshot_next)
        self.obs = outcomes[execute_index][0]
        self.last_executed_action = int(actions[execute_index])
        return outcomes

    def close(self):
        self.env.close()

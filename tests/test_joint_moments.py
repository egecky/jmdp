import unittest

from experiments_v2.envs import make_coupled_reward_chain, make_uniform_policy
from experiments_v2.jipe2_tabular import apply_T2, zero_moments


class JointMomentSmokeTest(unittest.TestCase):
    def test_shared_randomness_changes_cross_action_moment(self):
        kernel, n_states, n_actions, _ = make_coupled_reward_chain(nS=2)
        policy = make_uniform_policy(n_actions)
        moments = zero_moments(n_states, n_actions)

        shared = apply_T2(kernel, policy, gamma=0.9, M=moments, one_step_coupling=True)
        independent = apply_T2(kernel, policy, gamma=0.9, M=moments, one_step_coupling=False)

        self.assertAlmostEqual(shared.M_Sigma[0, 0, 0, 1], 0.0)
        self.assertAlmostEqual(independent.M_Sigma[0, 0, 0, 1], 0.25)


if __name__ == "__main__":
    unittest.main()

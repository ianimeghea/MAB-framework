import math
import random

from multi_agent_bandits.core.agent import Agent


class KLUCBAgent(Agent):
    """
    KL-UCB for Gaussian rewards (UCB-V / variance-adaptive variant).

    Standard UCB1 uses a fixed exploration bonus sqrt(2 log t / n). This
    is tight for Bernoulli rewards with variance 1/4 but loose when actual
    arm variance is lower. KL-UCB for Gaussian distributions tightens the
    bonus by incorporating the empirical variance:

        score(a) = μ̂(a) + sqrt(2 · (σ̂²(a) + prior_var) · log(t) / n(a))

    The +prior_var floor (default 1.0, matching the SD scale of the arms)
    prevents the bonus collapsing to zero after a single pull when the
    sample variance is still 0, and ensures exploration is at least as
    aggressive as UCB1 when variance is low.

    Effect: arms with high fill-rate volatility (e.g., a dark pool with
    inconsistent fill quality) receive a larger exploration bonus than
    stable arms with similar mean — matching the intuition that uncertain
    venues warrant more sampling before being ruled out.

    Symmetry-breaking: exploration order randomised at init.
    """

    def __init__(self, n_arms, prior_var=1.0, name=None):
        super().__init__(n_arms, name=name)
        self.prior_var = prior_var
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms        # running mean
        self.sq_values = [0.0] * n_arms     # running mean of r²  (for variance)
        self.t = 0
        self._explore_order = list(range(n_arms))
        random.shuffle(self._explore_order)
        self.last_arm = None

    def choose_arm(self):
        self.t += 1

        for arm in self._explore_order:
            if self.counts[arm] == 0:
                self.last_arm = arm
                return arm

        log_t = math.log(self.t)
        scores = []
        for arm in range(self.n_arms):
            n = self.counts[arm]
            mu = self.values[arm]
            # Sample variance: E[r²] - (E[r])²; clamped ≥ 0 for numerical safety
            var = max(self.sq_values[arm] - mu * mu, 0.0)
            bonus = math.sqrt(2.0 * (var + self.prior_var) * log_t / n)
            scores.append(mu + bonus)

        self.last_arm = max(range(self.n_arms), key=lambda a: scores[a])
        return self.last_arm

    def update(self, reward):
        arm = self.last_arm
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        self.sq_values[arm] += (reward * reward - self.sq_values[arm]) / n

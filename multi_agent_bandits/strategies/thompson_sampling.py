import math
import random

from multi_agent_bandits.core.agent import Agent


class ThompsonSamplingAgent(Agent):
    """
    Gaussian Thompson Sampling.

    Models each arm's reward as N(μ, σ²) with unknown mean. Maintains a
    Gaussian posterior over μ using a normal-normal conjugate update:

        prior:     μ ~ N(0, prior_var)
        posterior: μ | data ~ N(μ_n, τ_n²)
            τ_n² = 1 / (n + 1/prior_var)
            μ_n  = τ_n² * Σr_i  (collapsed from running mean)

    At each step, samples one draw from each arm's posterior and picks the
    arm with the highest sample. Exploration is implicit — uncertainty
    (wide posterior) produces high samples probabilistically.

    prior_var=100 gives a nearly uninformative prior relative to reward
    scales of 1–5, so the posterior is dominated by observations quickly.

    Symmetry-breaking: exploration order is randomised at init so that
    identical agents don't always pull the same unvisited arm first.
    """

    def __init__(self, n_arms, prior_var=100.0, name=None):
        super().__init__(n_arms, name=name)
        self.prior_var = prior_var
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms      # running posterior mean
        self._explore_order = list(range(n_arms))
        random.shuffle(self._explore_order)
        self.last_arm = None

    def choose_arm(self):
        samples = []
        for arm in range(self.n_arms):
            # Posterior variance shrinks as we observe more from this arm
            post_var = 1.0 / (self.counts[arm] + 1.0 / self.prior_var)
            samples.append(random.gauss(self.values[arm], math.sqrt(post_var)))
        self.last_arm = max(range(self.n_arms), key=lambda a: samples[a])
        return self.last_arm

    def update(self, reward):
        arm = self.last_arm
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

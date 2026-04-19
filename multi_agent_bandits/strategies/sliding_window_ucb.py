import math
import random
from collections import deque

from multi_agent_bandits.core.agent import Agent


class SlidingWindowUCBAgent(Agent):
    """
    Sliding-Window UCB for non-stationary bandits (SW-UCB).

    Only uses observations from the most recent `window` timesteps when
    computing arm estimates and exploration bonuses. This allows the agent
    to track drifting arm means — directly relevant to dark pool execution
    quality, which shifts intraday as liquidity migrates between venues.

    UCB index:
        score(a) = μ̂_w(a) + sqrt(ξ · log(min(t, W)) / n_w(a))

    where μ̂_w and n_w are the windowed mean and count for arm a.
    ξ=1.0 gives standard UCB1 confidence width within the window.

    Symmetry-breaking: exploration order is randomised at init.
    """

    def __init__(self, n_arms, window=200, xi=1.0, name=None):
        super().__init__(n_arms, name=name)
        self.window = window
        self.xi = xi
        # Each arm stores (timestep, reward) pairs within the sliding window
        self.history = [deque() for _ in range(n_arms)]
        self.t = 0
        self._explore_order = list(range(n_arms))
        random.shuffle(self._explore_order)
        self.last_arm = None

    def _prune(self, arm):
        """Discard observations that have fallen outside the window."""
        cutoff = self.t - self.window
        while self.history[arm] and self.history[arm][0][0] <= cutoff:
            self.history[arm].popleft()

    def choose_arm(self):
        self.t += 1
        for arm in range(self.n_arms):
            self._prune(arm)

        # Force exploration of arms with no in-window observations
        for arm in self._explore_order:
            if len(self.history[arm]) == 0:
                self.last_arm = arm
                return arm

        log_term = math.log(min(self.t, self.window))
        scores = []
        for arm in range(self.n_arms):
            h = self.history[arm]
            n_w = len(h)
            mean_w = sum(r for _, r in h) / n_w
            bonus = math.sqrt(self.xi * log_term / n_w)
            scores.append(mean_w + bonus)

        self.last_arm = max(range(self.n_arms), key=lambda a: scores[a])
        return self.last_arm

    def update(self, reward):
        self.history[self.last_arm].append((self.t, reward))

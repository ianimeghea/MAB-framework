import math
import random
from multi_agent_bandits.core.agent import Agent

class UCB_BaselineAgent(Agent):
    """
    Baseline UCB implementation.
    Chooses arm based on: estimated_value + exploration_bonus
    where exploration_bonus shrinks with more pulls
    """

    def __init__(self, n_arms, name = None):
        super().__init__(n_arms, name=name)

        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.total_steps = 0
        self.last_arm = None
        # Randomise the initial exploration order across agents.
        # This is a critical design choice for multi-agent experiments:
        # without it, all agents pull arms in the same order (0,1,...,K-1),
        # guaranteeing a collision at every exploration step and masking any
        # diversification signal. With it, agents take different exploration
        # paths, so subsequent specialisation (e.g. under zero_on_collision)
        # is driven by collision feedback — not by this randomisation.
        # The RQ2 result should be interpreted as: "given independent
        # exploration paths, collision feedback alone is sufficient to drive
        # venue separation." It is not a claim that UCB agents coordinate
        # strategically in general.
        self._explore_order = list(range(n_arms))
        random.shuffle(self._explore_order)

    def choose_arm(self):
        self.total_steps += 1

        #ensure we try each arm at least once
        for arm in self._explore_order:
            if self.counts[arm] == 0:
                self.last_arm = arm
                return arm


        ucb_scores = []
        for arm in range(self.n_arms):
            bonus = math.sqrt((2 * math.log(self.total_steps)) / self.counts[arm])
            ucb_scores.append(self.values[arm] + bonus)

        #pick best
        self.last_arm = max(range(self.n_arms), key=lambda a: ucb_scores[a])
        return self.last_arm

    def update(self, reward):
        arm = self.last_arm
        self.counts[arm] += 1
        step = 1 / self.counts[arm]
        self.values[arm] += step * (reward - self.values[arm])

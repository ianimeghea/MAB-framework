from multi_agent_bandits.core.agent import Agent


class FixedArmAgent(Agent):
    """
    Always routes to the same arm index.

    Models a static Smart Order Router that has been configured to always
    send orders to one specific dark pool — common in practice when
    institutions form persistent venue preferences based on prior analysis,
    broker relationships, or legacy infrastructure.

    Upper bound for static routing in stationary environments (if arm_idx
    happens to be the best arm); fails in non-stationary markets because it
    never re-evaluates the routing decision.
    """

    def __init__(self, n_arms, arm_idx=0, name=None):
        super().__init__(n_arms, name=name)
        self.arm_idx  = arm_idx
        self.last_arm = arm_idx

    def choose_arm(self):
        return self.arm_idx

    def update(self, reward):
        pass  # fixed routing — ignores execution feedback


class RoundRobinAgent(Agent):
    """
    Cycles through arms sequentially: 0, 1, 2, ..., n-1, 0, 1, ...

    Models equal-weight venue spreading — the SOR distributes order flow
    evenly across all pools without learning from execution quality. A
    common naive diversification baseline that satisfies 'must use all
    venues' mandates without any adaptive logic.
    """

    def __init__(self, n_arms, name=None):
        super().__init__(n_arms, name=name)
        self.t        = 0
        self.last_arm = 0

    def choose_arm(self):
        self.last_arm = self.t % self.n_arms
        self.t += 1
        return self.last_arm

    def update(self, reward):
        pass  # fixed rotation — ignores execution feedback

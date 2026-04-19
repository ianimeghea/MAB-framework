from multi_agent_bandits.core.environment import Environment
from multi_agent_bandits.core.experiment_runner import ExperimentRunner
from multi_agent_bandits.core.arm import Arm
from multi_agent_bandits.strategies.random import RandomAgent
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent
from multi_agent_bandits.strategies.ucb_baseline import UCB_BaselineAgent

def main(steps=1000, save_dir=None, plot_rewards=False, plot_frequencies=False):
    """
    Experiment 3: Strategy Comparison.
    Compares Random, Epsilon-Greedy, and UCB strategies in the same environment.
    """
    print("\n--- Running Experiment 3: Strategy Comparison ---")
    
    arms = [
        Arm(mean=1.0),
        Arm(mean=2.0),
        Arm(mean=3.0),
        Arm(mean=4.0)
    ]
    
    env = Environment(n_agents=3, arms=arms)
    
    agents = [
        RandomAgent(env.n_arms),
        EpsilonGreedyAgent(env.n_arms, epsilon=0.1),
        UCB_BaselineAgent(env.n_arms)
    ]
    
    runner = ExperimentRunner(
        env,
        agents,
        timestep_limit=steps,
        save_dir=save_dir
    )
    
    runner.run(
        plot_rewards=plot_rewards,
        plot_frequencies=plot_frequencies
    )
    
    runner.print_summary()

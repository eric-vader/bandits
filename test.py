import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import bandits as bd

n_arms = 10
n_trials = 1000
n_experiments = 500
bandit = bd.BernoulliBandit(n_arms, t=3*n_trials)  # cache samples ahead of time for speed

agents = [
    bd.Agent(bandit, bd.EpsilonGreedyPolicy(0.1)),
    bd.Agent(bandit, bd.UCBPolicy(1)),
    bd.BetaAgent(bandit, bd.GreedyPolicy())
]
env = bd.Environment(bandit, agents, label='Bayesian Bandits')
scores, optimal = env.run(n_trials, n_experiments)
env.plot_results(scores, optimal)
env.plot_beliefs()
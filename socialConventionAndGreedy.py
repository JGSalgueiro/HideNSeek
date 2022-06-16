import unittest
import numpy as np

from aasma.simplified_predator_prey import SimplifiedPredatorPrey

from aasma import GreedyAgent
from aasma import Agent, SocialConventionAgent
from randomAndGreedy import print_goodies
from randomVsRandom import run_multi_agent

class MyTestCase(unittest.TestCase):
    def test_GreedySeekerVsSocialConventionPrey(self):
        n_agents = 5

        # 1 - Setup environment
        environment = SimplifiedPredatorPrey(
            grid_shape=(20, 20),
            n_agents=5, n_preys=5,
            max_steps=100, required_captors=1
        )

        # 2 - Setup agent
        seekers = [GreedyAgent(i, environment.n_agents, environment.n_preys, False, environment) for i in
                   range(n_agents)]
        preys = [SocialConventionAgent(i, environment.n_agents, environment.n_preys, True, environment) for i in range(n_agents)]

        # 3 - Evaluate agent
        print_goodies(run_multi_agent(environment, seekers, preys, 20, lambda epi, step: epi % 5 == 0, 0.1))


    def test_SocialConventionSeekerVsGreedyPrey(self):
        n_agents = 5

        # 1 - Setup environment
        environment = SimplifiedPredatorPrey(
            grid_shape=(20, 20),
            n_agents=5, n_preys=5,
            max_steps=100, required_captors=1
        )

        # 2 - Setup agent
        seekers = [SocialConventionAgent(i, environment.n_agents, environment.n_preys, False, environment) for i in
                   range(n_agents)]
        preys = [GreedyAgent(i, environment.n_agents, environment.n_preys, True,
                                       environment) for i in range(n_agents)]

        # 3 - Evaluate agent
        print_goodies(run_multi_agent(environment, seekers, preys, 20, lambda epi, step: epi % 5 == 0, 0.1))


    def test_GreedySeekerVsSocialConventionPrey_ManyEpisodes(self):
        n_agents = 5

        # 1 - Setup environment
        environment = SimplifiedPredatorPrey(
            grid_shape=(20, 20),
            n_agents=5, n_preys=5,
            max_steps=100, required_captors=1
        )

        # 2 - Setup agent
        seekers = [GreedyAgent(i, environment.n_agents, environment.n_preys, False, environment) for i in
                   range(n_agents)]
        preys = [SocialConventionAgent(i, environment.n_agents, environment.n_preys, True, environment) for i in range(n_agents)]

        # 3 - Evaluate agent
        print_goodies(run_multi_agent(environment, seekers, preys, 500, lambda epi, step: False, 0.1))


    def test_SocialConventionSeekerVsGreedyPrey_ManyEpisodes(self):
        n_agents = 5

        # 1 - Setup environment
        environment = SimplifiedPredatorPrey(
            grid_shape=(20, 20),
            n_agents=5, n_preys=5,
            max_steps=100, required_captors=1
        )

        # 2 - Setup agent
        seekers = [SocialConventionAgent(i, environment.n_agents, environment.n_preys, False, environment) for i in
                   range(n_agents)]
        preys = [GreedyAgent(i, environment.n_agents, environment.n_preys, True,
                                       environment) for i in range(n_agents)]

        # 3 - Evaluate agent
        print_goodies(run_multi_agent(environment, seekers, preys, 500, lambda epi, step: False, 0.1))

if __name__ == '__main__':
    unittest.main()

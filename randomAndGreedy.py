import unittest

from aasma.simplified_predator_prey import SimplifiedPredatorPrey

from aasma import GreedyAgent
from aasma import Agent
from randomVsRandom import run_multi_agent

class MyTestCase(unittest.TestCase):
    def test_GreedySeekerVsRandomPrey(self):
        n_agents = 5

        # 1 - Setup environment
        environment = SimplifiedPredatorPrey(
            grid_shape=(20, 20),
            n_agents=5, n_preys=5,
            max_steps=100, required_captors=1
        )

        # 2 - Setup agent
        seekers = [GreedyAgent(i, environment.n_agents, environment.n_preys, False) for i in range(n_agents)]
        preys = [Agent(i, environment.n_agents, environment.n_preys, True) for i in range(n_agents)]

        # 3 - Evaluate agent
        print(run_multi_agent(environment, seekers, preys, 20, lambda epi, step: epi % 5 == 0, 0.1))


    def test_RandomSeekerVsGreedyPrey(self):
        n_agents = 5

        # 1 - Setup environment
        environment = SimplifiedPredatorPrey(
            grid_shape=(20, 20),
            n_agents=5, n_preys=5,
            max_steps=100, required_captors=1
        )

        # 2 - Setup agent
        seekers = [Agent(i, environment.n_agents, environment.n_preys, False) for i in range(n_agents)]
        preys = [GreedyAgent(i, environment.n_agents, environment.n_preys, True) for i in range(n_agents)]

        # 3 - Evaluate agent
        print(run_multi_agent(environment, seekers, preys, 20, lambda epi, step: epi % 5 == 0, 0.1))


    def test_GreedySeekerVsGreedyPrey(self):
        n_agents = 5

        # 1 - Setup environment
        environment = SimplifiedPredatorPrey(
            grid_shape=(20, 20),
            n_agents=5, n_preys=5,
            max_steps=100, required_captors=1
        )

        # 2 - Setup agent
        seekers = [GreedyAgent(i, environment.n_agents, environment.n_preys, False) for i in range(n_agents)]
        preys = [GreedyAgent(i, environment.n_agents, environment.n_preys, True) for i in range(n_agents)]

        # 3 - Evaluate agent
        print(run_multi_agent(environment, seekers, preys, 20, lambda epi, step: epi % 5 == 0, 0.1))


if __name__ == '__main__':
    unittest.main()
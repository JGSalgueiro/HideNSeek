import unittest

from aasma.simplified_predator_prey import SimplifiedPredatorPrey

from aasma import GreedyAgent
from aasma import Agent
from randomVsRandom import run_multi_agent

class MyTestCase(unittest.TestCase):
    def test_GreedySeekerVsRandomPrey(self):
        # 1 - Setup environment
        environment = SimplifiedPredatorPrey(
            grid_shape=(20, 20),
            n_agents=3, n_preys=3,
            max_steps=100, required_captors=1
        )

        # 2 - Setup agent
        seekers = [GreedyAgent(0, environment.n_agents, environment.n_preys, False),
                   GreedyAgent(1, environment.n_agents, environment.n_preys, False),
                   GreedyAgent(2, environment.n_agents, environment.n_preys, False)]
        preys = [Agent(0, environment.n_agents, environment.n_preys, True),
                 Agent(1, environment.n_agents, environment.n_preys, True),
                 Agent(2, environment.n_agents, environment.n_preys, True)]

        # 3 - Evaluate agent
        results = run_multi_agent(environment, seekers, preys, 10)

        print(results)


if __name__ == '__main__':
    unittest.main()
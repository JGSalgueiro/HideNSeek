import unittest

from aasma import MemoryAgent, NeuralDecentralizedVectorAgent, Agent, NeuralCentralizedVectorAgent
from aasma.simplified_predator_prey import SimplifiedPredatorPrey
from randomAndGreedy import print_goodies
from randomVsRandom import load_info, EpisodeInfo, run_multi_agent
from aasma.neuralNetworks import load_network


class MyTestCase(unittest.TestCase):
    def test_NeuralSeekerVsRandomPrey(self):
        n_agents = 5
        filenames = "Generation_10_Family_27_Seeker_"

        # 1 - Setup environment
        environment = SimplifiedPredatorPrey(
            grid_shape=(20, 20),
            n_agents=5, n_preys=5,
            max_steps=100, required_captors=1
        )

        # 2 - Setup agent
        networks = []

        for i in range(n_agents):
            filename = filenames + str(i)
            networks.append(load_network(filename))

        seekers = [NeuralDecentralizedVectorAgent(i, environment.n_agents, environment.n_preys,
                                                  False, environment, neuralNetwork=networks[i])
                   for i in range(n_agents)]
        preys = [Agent(i, environment.n_agents, environment.n_preys, True, environment) for i in range(n_agents)]

        # 3 - Evaluate agent
        print_goodies(run_multi_agent(environment, seekers, preys, 20, lambda epi, step: epi % 5 == 0, 0.1))

    def test_NeuralSeekerVsRandomPrey_ManyEpisodes(self):
        n_agents = 5
        filenames = "Generation_250_Family_4_Seeker_"

        # 1 - Setup environment
        environment = SimplifiedPredatorPrey(
            grid_shape=(20, 20),
            n_agents=5, n_preys=5,
            max_steps=100, required_captors=1
        )

        # 2 - Setup agent
        networks = []

        for i in range(n_agents):
            filename = filenames + str(i)
            networks.append(load_network(filename))

        seekers = [NeuralDecentralizedVectorAgent(i, environment.n_agents, environment.n_preys,
                                                  False, environment, neuralNetwork=networks[i])
                   for i in range(n_agents)]
        preys = [Agent(i, environment.n_agents, environment.n_preys, True, environment) for i in range(n_agents)]

        # 3 - Evaluate agent
        print_goodies(run_multi_agent(environment, seekers, preys, 500, lambda epi, step: False, 0.1))

    def test_CentralizedNeuralSeekerVsRandomPrey(self):
        n_agents = 5
        filename = "Family_19_Generation_290"

        # 1 - Setup environment
        environment = SimplifiedPredatorPrey(
            grid_shape=(20, 20),
            n_agents=5, n_preys=5,
            max_steps=100, required_captors=1
        )

        # 2 - Setup agent
        network = load_network(filename)

        seekers = [NeuralCentralizedVectorAgent(i, environment.n_agents, environment.n_preys,
                                                  False, environment, neuralNetwork=network, family_id=0)
                   for i in range(n_agents)]
        preys = [Agent(i, environment.n_agents, environment.n_preys, True, environment) for i in range(n_agents)]

        # 3 - Evaluate agent
        print_goodies(run_multi_agent(environment, seekers, preys, 20, lambda epi, step: epi % 5 == 0, 0.1))

    def test_NeuralSeekerVsRandomPrey_ManyEpisodes(self):
        n_agents = 5
        filenames = "Generation_250_Family_4_Seeker_"

        # 1 - Setup environment
        environment = SimplifiedPredatorPrey(
            grid_shape=(20, 20),
            n_agents=5, n_preys=5,
            max_steps=100, required_captors=1
        )

        # 2 - Setup agent
        networks = []

        for i in range(n_agents):
            filename = filenames + str(i)
            networks.append(load_network(filename))

        seekers = [NeuralDecentralizedVectorAgent(i, environment.n_agents, environment.n_preys,
                                                  False, environment, neuralNetwork=networks[i])
                   for i in range(n_agents)]
        preys = [Agent(i, environment.n_agents, environment.n_preys, True, environment) for i in range(n_agents)]

        # 3 - Evaluate agent
        print_goodies(run_multi_agent(environment, seekers, preys, 500, lambda epi, step: False, 0.1))


if __name__ == '__main__':
    unittest.main()

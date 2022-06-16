import unittest

from aasma import MemoryAgent
from aasma.simplified_predator_prey import SimplifiedPredatorPrey
from randomAndGreedy import print_goodies
from randomVsRandom import load_info, EpisodeInfo, run_multi_agent


class MyTestCase(unittest.TestCase):
    def test_MemorySeekerVsMemoryPrey(self):
        episode_filename = "neuralCentralized/Family_4_Generation_100"
        n_agents = 5

        # 1 - Setup environment
        environment = SimplifiedPredatorPrey(
            grid_shape=(20, 20),
            n_agents=5, n_preys=5,
            max_steps=100, required_captors=1
        )
        
        # 1.1 Fetch episode info
        episodeInfo = load_info(episode_filename)

        # 2 - Setup agent
        seekers = [MemoryAgent(i, environment.n_agents, environment.n_preys, False, environment, episodeInfo) for i in
                   range(n_agents)]
        preys = [MemoryAgent(i, environment.n_agents, environment.n_preys, True, environment, episodeInfo) for i in
                range(n_agents)]

        # 3 - Evaluate agent
        print_goodies(run_multi_agent(environment, seekers, preys, 1, lambda epi, step: True, 0.2,
                                      seekers_start_pos=episodeInfo.get_start_pos_seeker(),
                                      prey_start_pos=episodeInfo.get_start_pos_prey()))


if __name__ == '__main__':
    unittest.main()

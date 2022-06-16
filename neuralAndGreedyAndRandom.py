import unittest

import numpy as np
import os
import time

from aasma import GreedyAgent, Agent
from aasma.simplified_predator_prey import SimplifiedPredatorPrey
from randomAndGreedy import print_goodies
from randomVsRandom import run_multi_agent
from aasma import NeuralDecentralizedVectorAgent, clone, reproduce
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from aasma.neuralNetworks import save_info

class MyTestCase(unittest.TestCase):
    def test_NeuralSelfishVectorSeekerVsRandomPrey_CloneReproduction(self):
        with ThreadPoolExecutor(os.cpu_count()) as thread_pool:
            n_agents = 5
            n_max_generations = 400
            current_generation = 0
            all_results = {}
            np.random.seed(625)
            n_families = 32 # Must be a multiple of 2
            results_lock = Lock()
            best_results = []
            mutate_chance = 0.01

            family_threshold = n_families // 2

            # 1 - Setup environment

            # Making several environments so that there are no problems when multi-threading
            environments = [
                SimplifiedPredatorPrey(
                    grid_shape=(20, 20),
                    n_agents=5, n_preys=5,
                    max_steps=100, required_captors=1
                )
                for _ in range(n_families)
            ]

            environment = environments[0]

            # 2 - Setup agent
            seeker_families = [
                [NeuralDecentralizedVectorAgent(i, environment.n_agents, environment.n_preys, False, environment)
                 for i in
                 range(n_agents)] for _ in range(n_families)]
            preys = [Agent(i, environment.n_agents, environment.n_preys, True, environment) for i in
                     range(n_agents)]

            while current_generation <= n_max_generations:
                start_time = time.time()
                family_fitness = {}

                # Function that runs multiple episodes for a single family of agents
                def run_family(family_i):
                    #print("Running family ", family_i)

                    def render_when(epi: int, step: int) -> bool:
                        # return current_generation % 10 == 0 and epi % 5 == 0 and family_i == 0
                        return False

                    def save_when(epi) -> bool:
                        return current_generation % 10 == 0 and epi == 0

                    def give_save_name(epi) -> str:
                        return "neural/Family_" + str(family_i) + "_Generation_" + str(current_generation)

                    seeker_family = seeker_families[family_i]

                    # 3 - Evaluate agent
                    results = run_multi_agent(environments[family_i], seeker_family, preys, 10, render_when, 0.1,
                                              save_when, give_save_name)

                    familyFitness = fitness(results, False, n_agents)

                    # Saving the results and fitness
                    results_lock.acquire()
                    all_results[(current_generation, family_i)] = results
                    family_fitness[family_i] = familyFitness
                    results_lock.release()

                    for seeker in seeker_family:
                        seeker.game_ended()

                    #print("Concluded running family ", family_i)

                futures = [thread_pool.submit(run_family, i) for i in range(n_families)]

                for future in futures:
                    future.result()

                print("---Sorting families from generation ", current_generation, "--- Generation completed in ",
                      time.time() - start_time, "s---")

                # Ordering the families by fitness
                sorted_families = sorted([(fam, fit) for fam, fit in family_fitness.items()], key=lambda x: x[1],
                                         reverse=True)
                best_family_i = sorted_families[0][0]
                best_results.append(all_results[(current_generation, best_family_i)])

                # Saving the best neural network of each generation
                for agent_id in range(n_agents):
                    save_info(seeker_families[best_family_i][agent_id].neuralNetwork,
                              "Generation_" + str(current_generation) + "_Family_" + str(best_family_i) +
                              "_Seeker_" + str(agent_id))

                # Killing off the half worst and replacing with slightly mutated clones from the half best
                for i in range(family_threshold):
                    family_to_reproduce = seeker_families[sorted_families[i][0]]
                    family_to_die = seeker_families[sorted_families[i + family_threshold][0]]

                    for agent_i in range(n_agents):
                        newNetwork = clone(family_to_reproduce[agent_i].neuralNetwork)
                        newNetwork.mutate(mutate_chance)

                        family_to_die[agent_i].neuralNetwork = newNetwork

                print("---Results of best family: ", sorted_families[0][0], ", from generation ", current_generation, "---")
                print_goodies(best_results[current_generation])
                print("------------")

                current_generation += 1

            # All generations have ended

    def test_NeuralSelfishVectorSeekerVsRandomPrey_ParentReproduction(self):
        with ThreadPoolExecutor(os.cpu_count()) as thread_pool:
            n_agents = 5
            n_max_generations = 400
            current_generation = 0
            all_results = {}
            np.random.seed(625)
            n_families = 48 # Must be a multiple of 4
            results_lock = Lock()
            best_results = []
            mutate_chance = 0.01

            family_threshold = n_families // 2

            # 1 - Setup environment

            # Making several environments so that there are no problems when multi-threading
            environments = [
                SimplifiedPredatorPrey(
                    grid_shape=(20, 20),
                    n_agents=5, n_preys=5,
                    max_steps=100, required_captors=1
                )
                for _ in range(n_families)
            ]

            environment = environments[0]

            # 2 - Setup agent
            seeker_families = [
                [NeuralDecentralizedVectorAgent(i, environment.n_agents, environment.n_preys, False, environment)
                 for i in
                 range(n_agents)] for _ in range(n_families)]
            preys = [Agent(i, environment.n_agents, environment.n_preys, True, environment) for i in
                     range(n_agents)]

            while current_generation <= n_max_generations:
                start_time = time.time()
                family_fitness = {}

                # Function that runs multiple episodes for a single family of agents
                def run_family(family_i):
                    #print("Running family ", family_i)

                    def render_when(epi: int, step: int) -> bool:
                        # return current_generation % 10 == 0 and epi % 5 == 0 and family_i == 0
                        return False

                    def save_when(epi) -> bool:
                        return current_generation % 10 == 0 and epi == 0

                    def give_save_name(epi) -> str:
                        return "neural/Family_" + str(family_i) + "_Generation_" + str(current_generation)

                    seeker_family = seeker_families[family_i]

                    # 3 - Evaluate agent
                    results = run_multi_agent(environments[family_i], seeker_family, preys, 10, render_when, 0.1,
                                              save_when, give_save_name)

                    familyFitness = fitness(results, False, n_agents)

                    # Saving the results and fitness
                    results_lock.acquire()
                    all_results[(current_generation, family_i)] = results
                    family_fitness[family_i] = familyFitness
                    results_lock.release()

                    for seeker in seeker_family:
                        seeker.game_ended()

                    #print("Concluded running family ", family_i)

                futures = [thread_pool.submit(run_family, i) for i in range(n_families)]

                for future in futures:
                    future.result()

                print("---Sorting families from generation ", current_generation, "--- Generation completed in ",
                      time.time() - start_time, "s---")

                # Ordering the families by fitness
                sorted_families = sorted([(fam, fit) for fam, fit in family_fitness.items()], key=lambda x: x[1],
                                         reverse=True)
                best_family_i = sorted_families[0][0]
                best_results.append(all_results[(current_generation, best_family_i)])

                # Saving the best neural network of each generation
                for agent_id in range(n_agents):
                    save_info(seeker_families[best_family_i][agent_id].neuralNetwork,
                              "Generation_" + str(current_generation) + "_Family_" + str(best_family_i) +
                              "_Seeker_" + str(agent_id))

                # Killing off the half worst and replacing with slightly mutated clones from the half best
                for i in range(family_threshold):
                    mother_family = seeker_families[sorted_families[i // 2][0]]
                    father_family = seeker_families[sorted_families[i // 2 + 1][0]]
                    family_to_die = seeker_families[sorted_families[i + family_threshold][0]]

                    for agent_i in range(n_agents):
                        newNetwork = reproduce(mother_family[agent_i].neuralNetwork,
                                               father_family[agent_i].neuralNetwork)
                        newNetwork.mutate(mutate_chance)

                        family_to_die[agent_i].neuralNetwork = newNetwork

                print("---Results of best family: ", sorted_families[0][0], ", from generation ", current_generation, "---")
                print_goodies(best_results[current_generation])
                print("------------")

                current_generation += 1

            # All generations have ended

def fitness(results: tuple[np.ndarray, np.ndarray], is_prey: bool, n_prey: int):
    nRounds, nPreysAlive = results

    if is_prey:
        return nPreysAlive.mean()
    else:
        return n_prey - nPreysAlive.mean()

if __name__ == '__main__':
    unittest.main()

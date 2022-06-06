import argparse
from time import sleep

import numpy as np
from gym import Env
from typing import Sequence

from aasma import Agent
from aasma.utils import compare_results
from aasma.simplified_predator_prey import SimplifiedPredatorPrey
from aasma import greedyAgent

from collections.abc import Callable

def run_single_agent(environment: Env, agent: Agent, n_episodes: int) -> np.ndarray:

    results = np.zeros(n_episodes)

    for episode in range(n_episodes):

        steps = 0
        terminal = False
        observation = environment.reset()
        agent.see(observation)
        while not terminal:
            steps += 1
            action = agent.action()
            next_observation, reward, terminal, info = environment.step(action)
            agent.see(next_observation)
            environment.render()

        environment.close()

        results[episode] = steps

    return results

def run_multi_agent(environment: Env, seekers: Sequence[Agent], preys: Sequence[Agent], n_episodes: int,
                    render_when: Callable[[int, int], bool] = lambda episode, step: True,
                    seconds_per_rendered_frame: float = 0.05) -> np.ndarray:
    """
    Runs a multi agent system.
    environment -> the environment to run on
    seekers -> a list containing agents that will act as seekers
    preys -> a list containing agents that will act as prey
    n_episodes -> how many episodes should run in the given environment with the given agents
    render_when -> function that receives the number of the current episode and current step taking place and returns
        whether the game should be rendered
    seconds_per_rendered_frame -> amount of seconds per frame that gets rendered
    """
    results = np.zeros(n_episodes)

    for episode in range(n_episodes):

        steps = 0
        terminals = [False for _ in range(len(seekers))]
        observations = environment.reset()
        seekersActions = []
        preysActions = []

        if render_when(episode, steps):
            environment.render()

        while not all(terminals):
            steps += 1
            seekersActions.clear()
            preysActions.clear()
            for i in range(len(seekers)):
                seekers[i].receive_status(observations)
                seekersActions.append(seekers[i].action())

            for i in range(len(preys)):
                preys[i].receive_status(observations)
                preysActions.append(preys[i].action())
            
            next_observations, reward, terminals, info = environment.step(seekersActions, preysActions)

            if render_when(episode, steps):
                environment.render()
                print(episode, steps, next_observations, reward, terminals, info);
                sleep(seconds_per_rendered_frame)

            observations = next_observations
            

        results[episode] = steps

        environment.close()

    return results

# class RandomAgent(Agent):
#     n_actions = 0
#     def __init__(self, n_actions: int):
#         super(RandomAgent, self).__init__("Random Agent")
#         self.n_actions = n_actions

#     def action(self) -> int:
#         return np.random.randint(self.n_actions)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    opt = parser.parse_args()

    # 1 - Setup environment
    environment = SimplifiedPredatorPrey(
        grid_shape=(20, 20),
        n_agents=3, n_preys=3,
        max_steps=100, required_captors=1
    )

    # 2 - Setup agent
    agents = [Agent(0, environment.n_agents, environment.n_preys, False),
            Agent(1, environment.n_agents, environment.n_preys, False),
            Agent(2, environment.n_agents, environment.n_preys, False)]
    preys = [Agent(0, environment.n_agents, environment.n_preys, True),
            Agent(1, environment.n_agents, environment.n_preys, True),
            Agent(2, environment.n_agents, environment.n_preys, True)]

    # 3 - Evaluate agent
    results = run_multi_agent(environment, agents, preys, opt.episodes)

    # 4 - Compare results
    #compare_results(results, title="Random Agent on 'Predator Prey' Environment", colors=["orange"])


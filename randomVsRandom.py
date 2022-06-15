import argparse
from time import sleep

import numpy as np
import pickle
import os
from gym import Env
from typing import Sequence
from os.path import exists as file_exists

from aasma import Agent
from aasma.utils import compare_results
from aasma.simplified_predator_prey import SimplifiedPredatorPrey, ACTION_MEANING
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


def run_multi_agent(environment: SimplifiedPredatorPrey, seekers: Sequence[Agent], preys: Sequence[Agent], n_episodes: int,
                    render_when: Callable[[int, int], bool] = lambda episode, step: True,
                    seconds_per_rendered_frame: float = 0.05,
                    save_when: Callable[[int], bool] = lambda episode: False,
                    give_save_name: Callable[[int], str] = lambda episode: "Episode_" + str(episode),
                    seekers_start_pos: dict = None, prey_start_pos: dict = None) \
        -> tuple[np.ndarray, np.ndarray]:
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
    nRounds = np.zeros(n_episodes)
    nPreysAlive = np.zeros(n_episodes)

    for episode in range(n_episodes):

        steps = 0
        terminals = [False for _ in range(len(seekers))]
        observations = environment.reset()
        seekersActions = []
        preysActions = []
        episodeInfo = None

        if episode == 0:
            if seekers_start_pos is not None or prey_start_pos is not None:
                # At least one of the teams has pre-defined starting positions.
                # Clearing off the grid that was filled up with random positions due to the reset() method
                environment._full_obs = environment.create_grid()

                if seekers_start_pos is not None:
                    environment.agent_pos = seekers_start_pos
                if prey_start_pos is not None:
                    environment.prey_pos = prey_start_pos

                # Getting the agents placed on the grid
                for i in range(len(seekers)):
                    environment.update_agent_view(i)
                for i in range(len(preys)):
                    environment.update_prey_view(i)

        if render_when(episode, steps):
            environment.render()

        if save_when(episode):
            episodeInfo = EpisodeInfo()
            episodeInfo.initialize(len(seekers), len(preys), environment.agent_pos,
                                   environment.prey_pos, environment._max_steps)

        while not all(terminals):
            seekersActions.clear()
            preysActions.clear()

            for i in range(len(seekers)):
                seekers[i].receive_status(observations)
            for i in range(len(preys)):
                preys[i].receive_status(observations)

            for i in range(len(seekers)):
                seekersActions.append(seekers[i].action())
            for i in range(len(preys)):
                preysActions.append(preys[i].action())
                
            if save_when(episode):
                episodeInfo.add_step(steps, seekersActions, preysActions)

            next_observations, reward, terminals, info = environment.step(seekersActions, preysActions)

            if render_when(episode, steps):
                environment.render()
                """print(f'Epi: {episode}, Step: {steps}, Obs: {next_observations}, Reward: {reward}, Terminals: {terminals}, Info: {info}'
                      f'Seeker actions: {[ACTION_MEANING[a] for a in seekersActions]}, '
                      f'Prey actions: {[ACTION_MEANING[a] for a in preysActions]}')"""
                sleep(seconds_per_rendered_frame)

            observations = next_observations
            steps += 1

        nRounds[episode] = steps
        nPreysAlive[episode] = np.count_nonzero(info["prey_alive"])
        
        if save_when(episode):
            save_info(episodeInfo, give_save_name(episode))

        environment.close()

    return nRounds, nPreysAlive


# class RandomAgent(Agent):
#     n_actions = 0
#     def __init__(self, n_actions: int):
#         super(RandomAgent, self).__init__("Random Agent")
#         self.n_actions = n_actions

#     def action(self) -> int:
#         return np.random.randint(self.n_actions)

def to_np_array(start_pos: dict[int, list]):
    array = np.zeros(len(start_pos) * 2)

    for i in range(0, len(array), 2):
        array[i] = start_pos[i // 2][0]
        array[i + 1] = start_pos[i // 2][1]

    return array


def to_dict(start_pos: np.array):
    return {i: (int(start_pos[i * 2]), int(start_pos[i * 2 + 1])) for i in range(len(start_pos) // 2)}


class EpisodeInfo:
    def __init__(self):
        self.n_seekers = 0
        self.n_prey = 0
        self.start_pos_seekers = None
        self.start_pos_prey = None
        self.actions_seekers = None
        self.actions_prey = None

    def initialize(self, n_seekers: int, n_prey: int, start_pos_seekers: dict[int, list],
                   start_pos_prey: dict[int, list], steps: int):
        self.n_seekers = n_seekers
        self.n_prey = n_prey
        self.start_pos_seekers = to_np_array(start_pos_seekers)
        self.start_pos_prey = to_np_array(start_pos_prey)
        self.actions_seekers = np.zeros(n_seekers * steps * 2)
        self.actions_prey = np.zeros(n_prey * steps * 2)

    def add_step(self, step: int, actions_seekers: list[int], actions_prey: list[int]):
        step_offset = self.n_seekers * step
        for i in range(self.n_seekers):
            self.actions_seekers[step_offset + i] = actions_seekers[i]

        step_offset = self.n_prey * step
        for i in range(self.n_prey):
            self.actions_prey[step_offset + i] = actions_prey[i]

    def get_start_pos_seeker(self) -> dict:
        return to_dict(self.start_pos_seekers)

    def get_start_pos_prey(self) -> dict:
        return to_dict(self.start_pos_prey)

def load_info(filename: str) -> EpisodeInfo:
    filename = "savedEpisodes/" + filename + ".pickle"
    if file_exists(filename):
        file = open(filename, 'rb')
        content = pickle.load(file)
        file.close()

        return content
    else:
        raise FileNotFoundError(filename)


def save_info(info: EpisodeInfo, filename: str):
    os.makedirs("savedEpisodes/neural", exist_ok=True)
    file = open("savedEpisodes/" + filename + ".pickle", 'wb')
    pickle.dump(info, file)
    file.close()


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
    agents = [Agent(0, environment.n_agents, environment.n_preys, False, environment),
              Agent(1, environment.n_agents, environment.n_preys, False, environment),
              Agent(2, environment.n_agents, environment.n_preys, False, environment)]
    preys = [Agent(0, environment.n_agents, environment.n_preys, True, environment),
             Agent(1, environment.n_agents, environment.n_preys, True, environment),
             Agent(2, environment.n_agents, environment.n_preys, True, environment)]

    # 3 - Evaluate agent
    results = run_multi_agent(environment, agents, preys, opt.episodes)

    # 4 - Compare results
    # compare_results(results, title="Random Agent on 'Predator Prey' Environment", colors=["orange"])

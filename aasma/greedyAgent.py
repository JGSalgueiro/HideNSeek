import aasma.agent as agent
import numpy as np
import math
from scipy.spatial.distance import cityblock
import random


def _close_horizontally(distances):
    if distances[0] > 0:
        return agent.RIGHT
    elif distances[0] < 0:
        return agent.LEFT
    else:
        return agent.STAY


def _close_vertically(distances):
    if distances[1] > 0:
        return agent.DOWN
    elif distances[1] < 0:
        return agent.UP
    else:
        return agent.STAY


def direction_to_go(agent_position, prey_position):
    """
    Given the position of the agent and the position of a prey,
    returns the action to take in order to close the distance
    """

    distances = np.array(prey_position) - np.array(agent_position)
    abs_distances = np.absolute(distances)
    if abs_distances[0] > abs_distances[1]:
        return _close_horizontally(distances)
    elif abs_distances[0] < abs_distances[1]:
        return _close_vertically(distances)
    else:
        roll = random.uniform(0, 1)
        return _close_horizontally(distances) if roll > 0.5 else _close_vertically(distances)


def closest_prey(agent_position, prey_positions):
    """
    Given the positions of an agent and a sequence of positions of all prey,
    returns the positions of the closest prey.
    If there are no preys, None is returned instead
    """
    min = math.inf
    closest_prey_position = None
    n_preys = int(len(prey_positions) / 2)
    for p in range(n_preys):
        prey_position = prey_positions[p * 2], prey_positions[(p * 2) + 1]
        distance = cityblock(agent_position, prey_position)
        if distance < min:
            min = distance
            closest_prey_position = prey_position
    return closest_prey_position


class GreedyAgent(agent.Agent):
    def action(self) -> int:
        if len(self.visible_enemy_positions) == 0:
            # There are no visible enemies. Move randomly
            return np.random.randint(agent.N_ACTIONS)
        else:
            # There's at least 1 visible enemy
            closestPreyPosition =  closest_prey(self.current_position, self.visible_enemy_positions)
            direction = direction_to_go(self.current_position, closestPreyPosition)
            if self.is_prey():
                # Preys move away from the enemies
                return agent.OPPOSITE_ACTION[direction]
            else:
                # Seekers move towards the enemies
                return direction

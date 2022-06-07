import aasma.agent as agent
import numpy as np
import math
from scipy.spatial.distance import cityblock
import random


def closest_enemy(agent_position, enemy_positions):
    """
    Given the positions of an agent and a sequence of positions of all prey,
    returns the positions of the closest prey.
    If there are no preys, None is returned instead
    """
    min = math.inf
    closest_enemy_position = None
    n_preys = int(len(enemy_positions) / 2)
    for p in range(n_preys):
        enemy_position = enemy_positions[p * 2], enemy_positions[(p * 2) + 1]

        if enemy_position[0] == -100:
            # Prey is dead
            continue

        distance = cityblock(agent_position, enemy_position)
        if distance < min:
            min = distance
            closest_enemy_position = enemy_position
    return closest_enemy_position


class GreedyAgent(agent.Agent):
    def action(self) -> int:
        if len(self.visible_enemy_positions) == 0:
            if self.is_prey():
                return agent.STAY
            else:
                # There are no visible prey. Move randomly
                return np.random.randint(agent.N_ACTIONS)
        else:
            # There's at least 1 visible enemy
            closes_enemy_position = closest_enemy(self.current_position, self.visible_enemy_positions)

            return self.direction_to_go(self.current_position, closes_enemy_position)

    def _close_horizontally(self, agent_to_enemy, look_deeper=True):
        if self.is_prey():
            if agent_to_enemy[0] > 0:
                # Seeker is on the right
                if look_deeper:
                    # Returning either LEFT or another action in case LEFT leads to this agent
                    # not moving due to the grid's limits
                    return self.best_action(agent.LEFT, self._close_vertically(agent_to_enemy, False))
                else:
                    return agent.LEFT
            elif agent_to_enemy[0] < 0:
                # Seeker is on the left
                if look_deeper:
                    return self.best_action(agent.RIGHT, self._close_vertically(agent_to_enemy, False))
                else:
                    return agent.RIGHT
        else:
            if agent_to_enemy[0] > 0:
                # Prey is on the right
                return agent.RIGHT
            elif agent_to_enemy[0] < 0:
                # Prey is on the left
                return agent.LEFT

        return agent.STAY

    def _close_vertically(self, agent_to_enemy, look_deeper=True):
        if self.is_prey():
            if agent_to_enemy[1] > 0:
                # Seeker is downwards
                if look_deeper:
                    return self.best_action(agent.UP, self._close_horizontally(agent_to_enemy, False))
                else:
                    return agent.UP
            elif agent_to_enemy[1] < 0:
                # Seeker is upwards
                if look_deeper:
                    return self.best_action(agent.DOWN, self._close_horizontally(agent_to_enemy, False))
                else:
                    return agent.DOWN
        else:
            if agent_to_enemy[1] > 0:
                # Prey is downwards
                return agent.DOWN
            elif agent_to_enemy[1] < 0:
                # Prey is upwards
                return agent.UP

        return agent.STAY

    def direction_to_go(self, agent_position, enemy_position):
        """
        Given the position of the agent and the position of a prey,
        returns the action to take in order to close the distance
        """

        from_agent_to_enemy = np.array(enemy_position) - np.array(agent_position)
        abs_distances = np.absolute(from_agent_to_enemy)
        if abs_distances[0] > abs_distances[1]:
            return self._close_horizontally(from_agent_to_enemy)
        elif abs_distances[0] < abs_distances[1]:
            return self._close_vertically(from_agent_to_enemy)
        else:
            roll = random.uniform(0, 1)
            return self._close_horizontally(from_agent_to_enemy) if roll > 0.5 else self._close_vertically(
                from_agent_to_enemy)

    def best_action(self, *actions):
        """Checks which action out of the actions provided leads to
        the agent not being stuck in a wall. Chooses the 1st action that makes the agent stay inside
        the arena. (Ex: if RIGHT and UP are provided, then UP will be chosen if moving right leads to the
        agent not moving but moving up makes the agent move"""
        grid = self.environment.grid_shape

        for action in actions:
            movement = agent.ACTION_MOVEMENTS[action]
            next_position = self.current_position + movement

            # Checking if the next_position is inside the arena
            if 0 <= next_position[0] < grid[0] and 0 <= next_position[1] < grid[1]:
                return action

        return actions[0]

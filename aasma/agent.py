from abc import abstractmethod, ABC

import pygame
import random
import numpy as np

prey_view_range = 3
seeker_view_range = 2


class Agent(ABC):
    def __init__(self, row, col, is_prey: bool):
        self.team_positions = None
        self.visible_enemy_positions = None
        self.current_position = np.array((row, col))
        self._is_prey = is_prey
        self.eliminated = False
        self.direction = None
        self.view_range = prey_view_range if is_prey else seeker_view_range

    def is_prey(self):
        return self._is_prey

    def is_seeker(self):
        return not self._is_prey

    def eliminate(self):
        self.eliminated = True

    def move(self):
        decision = random.randint(0, 3)
        print(decision)

        if decision == 0:
            self.col = self.col - 1

        elif decision == 1:
            self.row = self.row + 1

        elif decision == 2:
            self.col = self.col + 1

        elif decision == 3:
            self.row = self.row - 1

    def receive_status(self, prey_positions: np.ndarray, seekers_positions: np.ndarray):
        """Receives the current status of the board and calls see() method with only what
        this agent can see"""
        if self.is_prey():
            team_positions = prey_positions
            enemy_positions = seekers_positions
        else:
            team_positions = seekers_positions
            enemy_positions = prey_positions

        visible_enemy_positions = []

        for i in range(len(enemy_positions)):
            # Filtering the enemy positions that this agent can see
            if (np.abs(enemy_positions[i] - self.current_position) <= self.view_range).all():
                visible_enemy_positions.append(enemy_positions[i])

        self.see(team_positions, np.array(visible_enemy_positions))

    def see(self, team_positions: np.ndarray, visible_enemy_positions: np.ndarray):
        self.team_positions = team_positions
        self.visible_enemy_positions = visible_enemy_positions

    @abstractmethod
    def action(self) -> int:
        raise NotImplementedError()

    def __repr__(self):
        return str("Prey" if self.is_prey() else "Seeker")

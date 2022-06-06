from abc import abstractmethod, ABC
from copyreg import add_extension

import pygame
import random
import numpy as np

PREY_VIEW_RANGE = 3
SEEKER_VIEW_RANGE = 2

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)
OPPOSITE_ACTION = {DOWN: UP, LEFT: RIGHT, UP: DOWN, RIGHT: LEFT, STAY: STAY}

class Agent(ABC):
    def __init__(self, agentId: int, nSeekers: int, nPreys: int, is_prey: bool):
        self.agentId = agentId
        self.nSeekers = nSeekers
        self.nPreys = nPreys
        self.team_positions = None
        self.visible_enemy_positions = None
        self.current_position = np.array((0,0))
        self._is_prey = is_prey
        self.eliminated = False
        self.direction = None
        self.view_range = PREY_VIEW_RANGE if is_prey else SEEKER_VIEW_RANGE

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
    
    def action(self) -> int:
        return np.random.randint(N_ACTIONS)

    def zipPairs(self, iterable):
        "s -> (s0, s1), (s2, s3), (s4, s5), ..."
        a = iter(iterable)
        return zip(a, a)

    def receive_status(self, observation: np.ndarray):
        """Receives the current status of the board and calls see() method with only what
        this agent can see"""

        if self.current_position[0] == -1 and self.current_position[1] == -1:
            return;

        seekers_positions = observation[:self.nSeekers * 2]
        prey_positions = observation[self.nSeekers * 2:]

        if self.is_prey():
            team_positions = prey_positions
            enemy_positions = seekers_positions
        else:
            team_positions = seekers_positions
            enemy_positions = prey_positions

        visible_enemy_positions = []
        
        self.current_position = np.array((team_positions[self.agentId * 2], team_positions[(self.agentId * 2) + 1]))

        for x,y in self.zipPairs(enemy_positions):
            # Filtering the enemy positions that this agent can see
            if (np.abs(x - self.current_position[0]) <= self.view_range) and (np.abs(y - self.current_position[1]) <= self.view_range):
                visible_enemy_positions.append(x)
                visible_enemy_positions.append(y)

        self.see(team_positions, np.array(visible_enemy_positions))

    def see(self, team_positions: np.ndarray, visible_enemy_positions: np.ndarray):
        self.team_positions = team_positions
        self.visible_enemy_positions = visible_enemy_positions

    # @abstractmethod
    # def action(self) -> int:
    #     raise NotImplementedError()

    def __repr__(self):
        return str("Prey" if self.is_prey() else "Seeker")
    
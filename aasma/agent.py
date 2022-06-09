import random
from abc import ABC
from time import sleep

import numpy as np

PREY_VIEW_RANGE = 3
SEEKER_VIEW_RANGE = 2

N_ACTIONS = 5
DOWN, LEFT, UP, RIGHT, STAY = range(N_ACTIONS)
OPPOSITE_ACTION = {DOWN: UP, LEFT: RIGHT, UP: DOWN, RIGHT: LEFT, STAY: STAY}
# Needed for Greedy Agent calculations
ACTION_MOVEMENTS = {DOWN: np.array((0, 1)), LEFT: np.array((-1, 0)),
                    UP: np.array((0, -1)), RIGHT: np.array((1, 0)), STAY: np.array((0, 0))}

class Agent(ABC):
    def __init__(self, agentId: int, nSeekers: int, nPreys: int, is_prey: bool, environment, wantsToReceiveInformation = False ):
        self.agentId = agentId
        self.nSeekers = nSeekers
        self.nPreys = nPreys
        self.team_positions = None
        self.visible_enemy_positions = None
        self.current_position = np.array((0, 0))
        self._is_prey = is_prey
        self.eliminated = False
        self.direction = None
        self.view_range = PREY_VIEW_RANGE if is_prey else SEEKER_VIEW_RANGE
        self.environment = environment
        self.wantsToShareInformation = wantsToReceiveInformation

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

        if self.current_position[0] == -100 and self.current_position[1] == -100:
            return

        for x,y in self.zipPairs(enemy_positions):
            # Filtering the enemy positions that this agent can see
            if (np.abs(x - self.current_position[0]) <= self.view_range) and (np.abs(y - self.current_position[1]) <= self.view_range):
                visible_enemy_positions.append(x)
                visible_enemy_positions.append(y)
        
        # print("isPrey: ", self.is_prey(), " agentId: ", self.agentId, "MyEnemyPositions: ", visible_enemy_positions)
        if self.wantsToShareInformation:
            for agentId in range(self.nPreys):
                if agentId == self.agentId:
                    continue
                self.calculateSharedInformation(enemy_positions, agentId, visible_enemy_positions, team_positions)

        # print("isPrey: ", self.is_prey(), " agentId: ", self.agentId, " realEnemyPositions:", visible_enemy_positions)
        # sleep(1)
        self.see(team_positions, np.array(visible_enemy_positions))

    def calculateSharedInformation(self, enemy_positions, agentId, visible_enemy_positions, team_positions):
        current_position = np.array((team_positions[agentId * 2], team_positions[(agentId * 2) + 1]))

        if current_position[0] == -100 and current_position[1] == -100:
            return

        for x,y in self.zipPairs(enemy_positions):
            # Filtering the enemy positions that this agent can see
            if (np.abs(x - current_position[0]) <= self.view_range) and (np.abs(y - current_position[1]) <= self.view_range):
                if not self.existsInList(x,y, visible_enemy_positions):
                    visible_enemy_positions.append(x)
                    visible_enemy_positions.append(y)

    def existsInList(self, xI, yI, visible_enemy_positions):
        for x,y in self.zipPairs(visible_enemy_positions):
            if x == xI and y == yI:
                return True
        return False

    def see(self, team_positions: np.ndarray, visible_enemy_positions: np.ndarray):
        self.team_positions = team_positions
        self.visible_enemy_positions = visible_enemy_positions

    # @abstractmethod
    # def action(self) -> int:
    #     raise NotImplementedError()

    def __repr__(self):
        return str("Prey" if self.is_prey() else "Seeker")
    
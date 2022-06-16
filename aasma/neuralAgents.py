import math
from abc import abstractmethod

import numpy as np

import aasma.agent as agent
import aasma.neuralNetworks as nn


class NeuralAgent(agent.Agent):

    def __init__(self, agentId: int, nSeekers: int, nPreys: int, is_prey: bool, environment,
                neuralNetwork: nn.NeuralNetwork = None, wantsToReceiveInformation=True):
        super(NeuralAgent, self).__init__(agentId, nSeekers, nPreys, is_prey, environment, wantsToReceiveInformation)

        self.neuralNetwork = None

        if neuralNetwork is None:
            self.set_empty_network()
        else:
            self.neuralNetwork = neuralNetwork

    @abstractmethod
    def turn_observation_into_neural_inputs(self) -> np.ndarray:
        raise NotImplementedError()

    def action(self) -> int:
        # By default, the action returned by "getAction" corresponds to the actual action with the same ID
        return self.neuralNetwork.getAction(self.turn_observation_into_neural_inputs())

    @abstractmethod
    def get_network_init_args(self) -> tuple[int, int, int, int]:
        raise NotImplementedError()

    def set_empty_network(self):
        self.neuralNetwork = self.get_emtpy_network()

    def get_emtpy_network(self):
        n_inputs, n_hidden_layers, n_hidden_layer_size, n_outputs = self.get_network_init_args()

        return nn.NeuralNetwork(n_inputs, n_hidden_layers, n_hidden_layer_size, n_outputs)

    def game_ended(self):
        pass

class NeuralDecentralizedVectorAgent(NeuralAgent):
    def __init__(self, agentId: int, nSeekers: int, nPreys: int, is_prey: bool, environment,
                neuralNetwork: nn.NeuralNetwork = None, wantsToReceiveInformation=True):
        self.visible_enemy_positions_history = []
        self.team_positions_history = []
        self.history_length = 3

        # Each input neuron will receive either the x or y value of a teammate or an enemy
        # There will be inputs for memorizing the last 2 states, hence the "* self.history_length"
        self.n_inputs = (nSeekers + nPreys) * 2 * self.history_length

        super().__init__(agentId, nSeekers, nPreys, is_prey, environment,
                         neuralNetwork, wantsToReceiveInformation)

    def turn_observation_into_neural_inputs(self) -> np.ndarray:
        self.visible_enemy_positions_history.append(self.visible_enemy_positions)
        self.team_positions_history.append(self.team_positions)

        # Deleting outdated history
        if len(self.visible_enemy_positions_history) > self.history_length:
            del self.visible_enemy_positions_history[0]
            del self.team_positions_history[0]

        inputs = np.zeros(self.n_inputs)
        n_inputs_per_history = self.n_inputs // self.history_length

        # Going through history backwards, since the last element is the most recent
        for his in range(len(self.visible_enemy_positions_history) - 1, -1, -1):
            visible_enemy_pos = self.visible_enemy_positions_history[his]
            team_pos = self.team_positions_history[his]
            next_his_offset = his * n_inputs_per_history

            # Setting this agent's position as the first inputs
            inputs[next_his_offset] = self.current_position[0]
            inputs[next_his_offset + 1] = self.current_position[1]

            # Setting the following inputs as the teammates' positions
            i = 2
            for agentId in range(self.nPreys if self.is_prey() else self.nSeekers):
                if self.agentId == agentId:
                    continue
                inputs[next_his_offset + i] = team_pos[agentId * 2]
                inputs[next_his_offset + i + 1] = team_pos[agentId * 2 + 1]

                i += 2

            # Setting the final inputs as the enemies' positions.

            enemy_pos_offset = next_his_offset + (self.nPreys if self.is_prey() else self.nSeekers) * 2

            # Going through all enemies
            for agentId in range(self.nSeekers if self.is_prey() else self.nPreys):
                if agentId * 2 >= len(visible_enemy_pos):
                    # There are no more visible enemies
                    inputs[enemy_pos_offset + agentId * 2] = -100
                    inputs[enemy_pos_offset + agentId * 2 + 1] = -100
                else:
                    # This enemy is visible
                    inputs[enemy_pos_offset + agentId * 2] = visible_enemy_pos[agentId * 2]
                    inputs[enemy_pos_offset + agentId * 2 + 1] = visible_enemy_pos[agentId * 2 + 1]

        return inputs

    def game_ended(self):
        # Resetting the history
        self.visible_enemy_positions_history = []
        self.team_positions_history = []

    def get_network_init_args(self) -> tuple[int, int, int, int]:
        # Each input neuron will receive either the x or y value of a teammate or an enemy
        # There will be inputs for memorizing the last 2 states, hence the "* self.history_length"
        n_inputs = self.n_inputs

        # No specific reason for this value
        n_hidden_layers = 1

        # Hopefully having double as many hidden neurons as input neurons will
        # make the agent recognize when preys are dead
        n_hidden_layer_size = 2 * n_inputs

        # As many outputs as actions the agent can take
        n_outputs = agent.N_ACTIONS

        return n_inputs, n_hidden_layers, n_hidden_layer_size, n_outputs

    def action(self) -> int:
        # Rotating the action, so that if getAction() returns 0, action() will return 4 (STAY), since 0 is what empty networks return
        return (self.neuralNetwork.getAction(self.turn_observation_into_neural_inputs()) + agent.N_ACTIONS - 1) % agent.N_ACTIONS

class NeuralCentralizedVectorAgent(NeuralAgent):
    family_network: dict[int, nn.NeuralNetwork] = {}
    family_visible_enemy_positions_history: dict[int, list] = {}
    family_team_positions_history: dict[int, list] = {}
    family_members: dict[int, list[int]] = {}
    last_step_inputs: dict[int, int] = {}
    calculated_inputs: dict[int, np.array] = {}
    last_step_prediction: dict[int, int] = {}
    calculated_prediction: dict[int, np.array] = {}

    def __init__(self, agentId: int, nSeekers: int, nPreys: int, is_prey: bool, environment,
                family_id: int, neuralNetwork: nn.NeuralNetwork = None, wantsToReceiveInformation=True):
        # Setting these variables right now since get_empty_network() needs it before it's assigned
        self._is_prey = is_prey
        self.nSeekers = nSeekers
        self.nPreys = nPreys

        self.history_length = 3

        self.family_id = family_id

        NeuralCentralizedVectorAgent.family_visible_enemy_positions_history.setdefault(family_id, [])
        NeuralCentralizedVectorAgent.family_team_positions_history.setdefault(family_id, [])
        NeuralCentralizedVectorAgent.family_members.setdefault(family_id, [])
        NeuralCentralizedVectorAgent.last_step_inputs.setdefault(family_id, -1)
        NeuralCentralizedVectorAgent.last_step_prediction.setdefault(family_id, -1)

        # Each input neuron will receive either the x or y value of a teammate or an enemy
        # There will be inputs for memorizing the last 2 states, hence the "* self.history_length"
        self.n_inputs = (nSeekers + nPreys) * 2 * self.history_length

        if family_id not in NeuralCentralizedVectorAgent.family_network:
            if neuralNetwork is None:
                neuralNetwork = self.get_emtpy_network()
                NeuralCentralizedVectorAgent.family_network[family_id] = self.get_emtpy_network()
            else:
                NeuralCentralizedVectorAgent.family_network[family_id] = neuralNetwork
        else:
            # Setting this variable so the super constructor doesn't create a new network
            neuralNetwork = NeuralCentralizedVectorAgent.family_network[family_id]

        NeuralCentralizedVectorAgent.family_members[family_id].append(agentId)

        super().__init__(agentId, nSeekers, nPreys, is_prey, environment,
                         neuralNetwork, wantsToReceiveInformation)

    def get_my_network(self) -> nn.NeuralNetwork:
        return NeuralCentralizedVectorAgent.family_network[self.family_id]

    def set_my_network(self, network: nn.NeuralNetwork):
        NeuralCentralizedVectorAgent.family_network[self.family_id] = network

    def turn_observation_into_neural_inputs(self) -> np.ndarray:
        current_step = self.environment._step_count

        if current_step > NeuralCentralizedVectorAgent.last_step_inputs[self.family_id]:
            # This is the first agent turning the observations into neural inputs
        
            NeuralCentralizedVectorAgent.last_step_inputs[self.family_id] += 1

            visible_enemy_positions_history = NeuralCentralizedVectorAgent.family_visible_enemy_positions_history[self.family_id]
            team_positions_history = NeuralCentralizedVectorAgent.family_team_positions_history[self.family_id]

            visible_enemy_positions_history.append(self.visible_enemy_positions)
            team_positions_history.append(self.team_positions)

            # Deleting outdated history
            if len(visible_enemy_positions_history) > self.history_length:
                del visible_enemy_positions_history[0]
                del team_positions_history[0]

            inputs = np.zeros(self.n_inputs)
            n_inputs_per_history = self.n_inputs // self.history_length

            # Going through history backwards, since the last element is the most recent
            for his in range(len(visible_enemy_positions_history) - 1, -1, -1):
                visible_enemy_pos = visible_enemy_positions_history[his]
                team_pos = team_positions_history[his]
                next_his_offset = his * n_inputs_per_history

                # Setting the following inputs as the teammates' positions
                for agentId in range(self.nPreys if self.is_prey() else self.nSeekers):
                    inputs[next_his_offset + agentId * 2] = team_pos[agentId * 2]
                    inputs[next_his_offset + agentId * 2 + 1] = team_pos[agentId * 2 + 1]

                # Setting the final inputs as the enemies' positions.

                enemy_pos_offset = next_his_offset + (self.nPreys if self.is_prey() else self.nSeekers) * 2

                # Going through all enemies
                for agentId in range(self.nSeekers if self.is_prey() else self.nPreys):
                    if agentId * 2 >= len(visible_enemy_pos):
                        # There are no more visible enemies
                        inputs[enemy_pos_offset + agentId * 2] = -100
                        inputs[enemy_pos_offset + agentId * 2 + 1] = -100
                    else:
                        # This enemy is visible
                        inputs[enemy_pos_offset + agentId * 2] = visible_enemy_pos[agentId * 2]
                        inputs[enemy_pos_offset + agentId * 2 + 1] = visible_enemy_pos[agentId * 2 + 1]

            NeuralCentralizedVectorAgent.calculated_inputs[self.family_id] = inputs

        return NeuralCentralizedVectorAgent.calculated_inputs[self.family_id]

    def game_ended(self):
        # Resetting the history
        NeuralCentralizedVectorAgent.family_visible_enemy_positions_history[self.family_id].clear()
        NeuralCentralizedVectorAgent.family_team_positions_history[self.family_id].clear()
        NeuralCentralizedVectorAgent.last_step_inputs[self.family_id] = -1
        NeuralCentralizedVectorAgent.last_step_prediction[self.family_id] = -1

    def get_network_init_args(self) -> tuple[int, int, int, int]:
        # Each input neuron will receive either the x or y value of a teammate or an enemy
        # There will be inputs for memorizing the last 2 states, hence the "* self.history_length"
        n_inputs = self.n_inputs

        # No specific reason for this value
        n_hidden_layers = 2

        # No specific reason for this value
        n_hidden_layer_size = 16

        # As many outputs as permutations of action pairs
        n_outputs = self.nPreys ** agent.N_ACTIONS if self.is_prey() else self.nSeekers ** agent.N_ACTIONS

        return n_inputs, n_hidden_layers, n_hidden_layer_size, n_outputs

    def action(self) -> int:
        current_step = self.environment._step_count

        if current_step > NeuralCentralizedVectorAgent.last_step_prediction[self.family_id]:
            # This is the first agent generating an action
            NeuralCentralizedVectorAgent.last_step_prediction[self.family_id] += 1
            NeuralCentralizedVectorAgent.calculated_prediction[self.family_id] = \
                self.get_my_network().getAction(self.turn_observation_into_neural_inputs())

        prediction = NeuralCentralizedVectorAgent.calculated_prediction[self.family_id]

        # This prediction value corresponds to the index of the collective action
        # Ex: nAgents = 5, nActions = 5
        #       Possible actions: 0: (0,0,0,0,0), 1: (0,0,0,0,1), 2: (0,0,0,0,2), ..., 12: (0,0,0,2,2)
        # Furthermore, we must return the action of this agent, so if prediction = 12 and agentID = 1
        # 2 will be returned
        # The general formula is: ((i // 625) % 5, (i // 125) % 5, (i // 25) % 5, (i // 5) % 5, i % 5),
        # where "i" is the prediction

        action = (prediction // (agent.N_ACTIONS ** self.agentId)) % agent.N_ACTIONS

        # Rotating the action, so that if getAction() returns 0, action() will return 4 (STAY), since 0 is what empty networks return
        return (action + agent.N_ACTIONS - 1) % agent.N_ACTIONS

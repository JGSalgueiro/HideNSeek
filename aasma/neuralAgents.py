from abc import abstractmethod

import cupy as np

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
        # Creating a neural network

        n_inputs, n_hidden_layers, n_hidden_layer_size, n_outputs = self.get_network_init_args()

        self.neuralNetwork = nn.NeuralNetwork(n_inputs, n_hidden_layers, n_hidden_layer_size, n_outputs)

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
            agentId = 0
            for agentId in range(self.nPreys if self.is_prey() else self.nSeekers):
                if self.agentId == agentId:
                    continue
                inputs[next_his_offset + i] = team_pos[agentId * 2]
                inputs[next_his_offset + i + 1] = team_pos[agentId * 2 + 1]

                i += 2

            # Setting the final inputs as the enemies' positions.

            enemy_pos_offset = next_his_offset + (self.nPreys if self.is_prey() else self.nSeekers) * 2

            # Going through all enemies
            for i in range(0, self.nSeekers if self.is_prey() else self.nPreys, 2):
                if i * 2 >= len(visible_enemy_pos):
                    # There are no more visible enemies
                    inputs[enemy_pos_offset + i] = -100
                    inputs[enemy_pos_offset + i + 1] = -100
                else:
                    # This enemy is visible
                    inputs[enemy_pos_offset + i] = visible_enemy_pos[i]
                    inputs[enemy_pos_offset + i + 1] = visible_enemy_pos[i + 1]

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
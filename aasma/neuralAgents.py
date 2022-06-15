from abc import abstractmethod

import aasma.agent as agent
import numpy as np
import math
from scipy.spatial.distance import cityblock
import random
import aasma.neuralNetworks as nn

class NeuralAgent(agent.Agent):

    def __int__(self, agentId: int, nSeekers: int, nPreys: int, is_prey: bool, environment,
                neuralNetwork: nn.NeuralNetwork, wantsToReceiveInformation = False):
        super(NeuralAgent, self).__init__(agentId, nSeekers, nPreys, is_prey, environment, wantsToReceiveInformation)
        self.neuralNetwork = neuralNetwork

    @abstractmethod
    def turn_observation_into_neural_inputs(self) -> np.ndarray:
        raise NotImplemented()

    def action(self) -> int:
        return self.neuralNetwork.getAction(self.turn_observation_into_neural_inputs())

class NeuralSelfishVectorAgent(NeuralAgent):
    def __int__(self, agentId: int, nSeekers: int, nPreys: int, is_prey: bool, environment,
                neuralNetwork: nn.NeuralNetwork):
        super(NeuralSelfishVectorAgent, self).__init__(agentId, nSeekers, nPreys, is_prey, environment, neuralNetwork, False)

    def turn_observation_into_neural_inputs(self) -> np.ndarray:
        raise NotImplemented()

    def set_empty_network(self):
        #

        #self.neuralNetwork = nn.NeuralNetwork()
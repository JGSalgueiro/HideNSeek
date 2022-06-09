import numpy as np
import random

weightMutationNoise = 2
percentageWeightsMutated = 0.01

class NeuralNetwork:
    def __init__(self, n_inputs: int, n_hidden_layers: int, n_hidden_layer_size: int, n_outputs: int,
                 initialize: bool = True):
        #TODO Add Bias
        if initialize:
            self.input_weights = self.initialize_matrix(n_hidden_layer_size, n_inputs)
            self.hidden_layers_weights = self.initialize_tensor(n_hidden_layers, n_hidden_layer_size,
                                                                n_hidden_layer_size)
            self.output_weights = self.initialize_matrix(n_outputs, n_hidden_layer_size)
        else:
            self.input_weights = None
            self.hidden_layers_weights = None
            self.output_weights = None

    def initialize_matrix(self, n_rows: int, n_columns: int):
        # We might want to create specializations where the matrices are not initialized with 0s
        return np.zeros((n_rows, n_columns))

    def initialize_tensor(self, x: int, y: int, z: int):
        # We might want to create specializations where the tensors are not initialized with 0s
        return np.zeros((x, y, z))

    def activation(self, x: float):
        # Sigmoid by default
        return 1.0 / (1.0 + np.exp(-1 * x))

    def try_to_mutate(self, chance):
        if np.random.random_sample() < chance:
            self.mutate()

    def mutate(self):
        # TODO
        pass

class RandomInitNeuralNetwork(NeuralNetwork):
    def initialize_matrix(self, n_rows: int, n_columns: int):
        array = np.random.rand(n_rows, n_columns)

        return array * 2 - np.ones((n_rows, n_columns))

    def initialize_tensor(self, x: int, y: int, z: int):
        array = np.random.rand(x, y, z)

        return array * 2 - np.ones((x, y, z))


def getEmptyNetwork() -> NeuralNetwork:
    return NeuralNetwork(0, 0, 0, 0, False)


def clone(original: NeuralNetwork) -> NeuralNetwork:
    copy = getEmptyNetwork()

    copy.input_weights = np.copy(original.input_weights)
    copy.hidden_layers_weights = np.copy(original.hidden_layers_weights)
    copy.output_weights = np.copy(original.output_weights)

    return copy

def reproduce(mother: NeuralNetwork, father: NeuralNetwork) -> NeuralNetwork:
    child = clone(mother)

    inputsFromFather = np.random.choice(np.arange(len(mother.input_weights)))

    #TODO
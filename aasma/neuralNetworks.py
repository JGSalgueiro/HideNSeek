import numpy as np
import random
import pickle
import os
from os.path import exists as file_exists

weightMutationNoise = 2
percentageWeightsMutated = 0.01

class NeuralNetwork:
    def __init__(self, n_inputs: int, n_hidden_layers: int, n_hidden_layer_size: int, n_outputs: int,
                 initialize: bool = True):
        if initialize:
            if n_hidden_layers >= 1:
                # There's at least 1 hidden layer, meaning the inputs do not directly connect to the outputs
                self.input_weights = self.initialize_matrix(n_hidden_layer_size, n_inputs + 1)
                self.output_weights = self.initialize_matrix(n_outputs, n_hidden_layer_size + 1)

                if n_hidden_layers >= 2:
                    # At least 2 hidden layers
                    self.hidden_layers_weights = self.initialize_tensor(n_hidden_layers - 1, n_hidden_layer_size,
                                                                        n_hidden_layer_size + 1)
                else:
                    # There's only 1 hidden layer
                    self.hidden_layers_weights = None
            else:
                # The input neurons directly connect to the outputs
                self.input_weights = self.initialize_matrix(n_outputs, n_inputs + 1)
                self.output_weights = None
                self.hidden_layers_weights = None

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

    def activation(self, x: np.array):
        # Sigmoid by default
        return sigmoid(x)

    def mutate(self, weight_mutate_chance):
        """Makes this network's weights change should the odds let them.
        The provided percentage measures how likely it is that a weight will be mutated.
        A mutation will change a weight's value """
        for weight_matrix in self.weight_matrix_sequence():
            for weight_vector in weight_matrix:
                # Getting how many weights will receive a mutation
                size = len(weight_vector)
                weights_to_change = np.random.binomial(size, weight_mutate_chance)

                if weights_to_change > 0:
                    weight_indexes = np.random.choice(np.arange(size), weights_to_change, False)
                    additions = np.random.random(weights_to_change) * weightMutationNoise - 1
                    weight_vector[weight_indexes] += additions

    def weight_matrix_sequence(self):
        yield self.input_weights

        if self.hidden_layers_weights is not None:
            for hidden_weight in self.hidden_layers_weights:
                yield hidden_weight

        if self.output_weights is not None:
            yield self.output_weights

    def predict(self, inputs: np.array) -> tuple[np.array, int]:
        """Receives an array of input values and returns
        a tuple containing all the outputs and the index of the
        output with the greatest value, which corresponds to the
        best action to take, should this be used for the SimplePredatorPrey game"""

        # Inputs format: 1d array of size n_inputs
        # inputs = self.activation(np.array(inputs))

        # Must be changed to 2d array of size (n_inputs + 1, 1) to do the math
        inputs = np.concatenate((inputs, np.ones([1]))).reshape((len(inputs) + 1, 1))

        for weight_matrix in self.weight_matrix_sequence():
            aux = self.activation(weight_matrix @ inputs)

            # Adding a row with a '1' due to the bias multiplication
            inputs = np.concatenate((aux, np.ones([1, 1])), axis=0)

        # Ignoring the last row because of the bias
        outputs = inputs[:-1]

        return outputs, np.argmax(outputs)

    def getAction(self, inputs: np.array) -> int:
        return self.predict(inputs)[1]

class RandomInitNeuralNetwork(NeuralNetwork):
    """NeuralNetwork whose weights and bias are initialized with random values
    between -1 and 1"""
    def initialize_matrix(self, n_rows: int, n_columns: int):
        array = np.random.rand(n_rows, n_columns)

        aux = array * 2 - np.ones((n_rows, n_columns))

        # Making all biases 0
        aux[:, -1] = 0

        return aux

    def initialize_tensor(self, x: int, y: int, z: int):
        array = np.random.rand(x, y, z)

        aux = array * 2 - np.ones((x, y, z))

        # Making all biases 0
        aux[:, :, -1] = 0

        return aux


def getEmptyNetwork() -> NeuralNetwork:
    return NeuralNetwork(0, 0, 0, 0, False)


def clone(original: NeuralNetwork) -> NeuralNetwork:
    copy = getEmptyNetwork()

    copy.input_weights = np.copy(original.input_weights)

    if original.hidden_layers_weights is not None:
        copy.hidden_layers_weights = np.copy(original.hidden_layers_weights)
    if original.output_weights is not None:
        copy.output_weights = np.copy(original.output_weights)

    return copy

def reproduce(mother: NeuralNetwork, father: NeuralNetwork) -> NeuralNetwork:
    """Creates a NeuralNetwork whose neurons were randomly selected either from
    its mother or its father"""
    child = clone(mother)

    moreWeights = True
    childWeightSequence = child.weight_matrix_sequence() # Yields child's input, hidden and output weights
    fatherWeightSequence = father.weight_matrix_sequence()

    # Replacing some weights inherited from the mother to weights inherited from the father
    while moreWeights:
        try:
            childWeight = next(childWeightSequence)
            fatherWeight = next(fatherWeightSequence)

            size = len(childWeight)

            # Getting random indexes corresponding to the weights that will be inherited by the father
            indexesFromFather = np.random.choice(np.arange(size), np.random.randint(0, size), False)

            childWeight[indexesFromFather] = fatherWeight[indexesFromFather]

        except StopIteration:
            moreWeights = False

    return child


def sigmoid(x: np.array):
    return 1.0 / (1.0 + np.exp(-1 * x))


def load_network(filename: str) -> NeuralNetwork:
    filename = "neuralNetworks/" + filename + ".pickle"
    if file_exists(filename):
        file = open(filename, 'rb')
        content = pickle.load(file)
        file.close()

        return content
    else:
        raise FileNotFoundError(filename)


def save_network(info: NeuralNetwork, filename: str):
    os.makedirs("neuralNetworks/", exist_ok=True)
    file = open("neuralNetworks/" + filename + ".pickle", 'wb')
    pickle.dump(info, file)
    file.close()
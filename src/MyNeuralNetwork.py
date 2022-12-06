import numpy as np


class MyNeuralNetwork:

    def __init__(self, inputs: np.array, targets: np.array, loss_fun=None):
        self.inputs = inputs
        self.targets = targets
        self.observations = len(inputs)
        if not loss_fun:
            loss_fun = lambda delta: np.sum(delta ** 2) / (2 * self.observations)
        self.loss_fun = loss_fun

    def apply_algth(self, learning_rate, iterations):
        weights = np.random.uniform(low=-1, high=1, size=(self.inputs.shape[1], 1))
        biases = np.random.uniform(low=-1, high=1, size=1)
        loss = None
        for i in range(iterations):
            outputs = np.dot(self.inputs, weights) + biases
            deltas = outputs - self.targets
            loss = self.loss_fun(deltas)
            deltas_scaled = deltas / self.observations
            weights = weights - learning_rate * np.dot(self.inputs.T, deltas_scaled)
            biases = biases - learning_rate * np.sum(deltas_scaled)
            if loss < 0.001:
                return weights, biases, loss

        return weights, biases, loss

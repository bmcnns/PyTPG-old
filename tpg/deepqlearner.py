import numpy as np

class DeepQLearner:

    def __init__(self, numStates, numActions, hiddenUnits, gamma, learningRate):
        rng = np.random.default_rng()
        self.numStates = numStates
        self.numActions = numActions
        self.hiddenUnits = hiddenUnits
        self.weights1 = rng.normal(0, 1, size=(self.numStates+1, self.hiddenUnits))
        self.weights2 = rng.normal(0, 1, size=(self.hiddenUnits+1, self.numActions))
        self.gamma = gamma
        self.learningRate = learningRate

    def relu(self, x):
        return np.maximum(0, x)

    def train(self, previousState, nextState, reward, action):
        previousInput = np.zeros(self.numStates + 1)
        previousInput[0] = -1
        previousInput[1:] = previousState

        nextInput = np.zeros(self.numStates + 1)
        nextInput[0] = -1
        nextInput[1:] = nextState

        # Calculate hidden layer activations using ReLU
        hidden_activations = np.dot(self.weights1.T, previousInput)
        hidden_activations = np.append(hidden_activations, -1)  # Bias term
        hidden_output = self.relu(hidden_activations)

        # Calculate the current Q-value
        currentQ = np.dot(self.weights2.T, hidden_output)

        # Calculate hidden layer activations for the next state using ReLU
        hidden_activations_next = np.dot(self.weights1.T, nextInput)
        hidden_activations_next = np.append(hidden_activations_next, -1)  # Bias term
        hidden_output_next = self.relu(hidden_activations_next)

        # Calculate the target Q-value using Q-learning
        maxNextQ = np.max(np.dot(self.weights2.T, hidden_output_next))
        targetQ = currentQ.copy()
        targetQ[action] = reward + self.gamma * maxNextQ

        error = targetQ - currentQ

        # Backpropagation
        hidden_delta = np.where(hidden_output > 0, 1, 0) * np.dot(self.weights2, error)
        self.weights1 += self.learningRate * np.outer(previousInput, hidden_delta[:-1])
        self.weights2 += self.learningRate * np.outer(hidden_output, error)

    def predict(self, state):
        input = np.zeros(self.numStates + 1)
        input[0] = -1
        input[1:] = state

        # Calculate hidden layer activations using ReLU
        hidden_activations = np.dot(self.weights1.T, input)
        hidden_activations = np.append(hidden_activations, -1)  # Bias term
        hidden_output = self.relu(hidden_activations)

        return np.dot(self.weights2.T, hidden_output)
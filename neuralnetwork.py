import numpy as np
import random as rd

def ReLU(x, derivative = False):
    if(derivative):
        return 1. * (x > 0)
    return x * (x > 0)

def sigmoid (x, derivative = False):
    if(derivative):
        return x * (1 - x)
    return 1/(1 + np.exp(-x))
    
class NeuralNetwork():
    def __init__(self, neurons):
        self.neurons = neurons
        self.weights = []
        self.bias = []
        previous_neuron = neurons[0]
        for neuron in neurons[1:]:
            weight = np.random.uniform(size=(previous_neuron, neuron))
            bia = np.random.uniform(size=(1, neuron))
            self.weights.append(weight)
            self.bias.append(bia)
            previous_neuron = neuron

    def predict(self, data_input, function_activation):
        predictions = []
        for i in range(len(self.weights)):
            prediction = function_activation(data_input.dot(self.weights[i]) + self.bias[i])
            predictions.append(prediction)
            data_input = prediction
        return predictions
    
    def calculateError(self, predictions, data_output):
        predictions_error = []
        predictions_error.append(data_output - predictions[-1])
        for i in range(len(predictions)-1):
            prediction_error = predictions_error[-1-i].dot(self.weights[-1-i].T)
            predictions_error.insert(0, prediction_error)
        return predictions_error

    def calculateGradient(self, predictions_error, predictions, learning_rate, function_activation):
        gradients = []
        for i in range(len(self.neurons)-1):
            gradient = predictions_error[i] * function_activation(predictions[i], derivative = True)
            gradient *= learning_rate
            gradients.append(gradient)
        return gradients
    def calculateDeltaWeights(self, predictions, gradients, data_input):
        deltaWeights = []
        for i in range(len(self.weights)):
            deltaWeight = data_input.T.dot(gradients[i])
            deltaWeights.append(deltaWeight)
            data_input = predictions[i]
        return deltaWeights

    def MSE(self, data_inputs, data_outputs, function_activation):
        soma = np.array([[0.0]])
        for i in range(len(data_inputs)):
            prediction = self.predict(np.array(data_inputs[i]), function_activation)[-1]
            pa = (np.array(data_outputs[i]) - np.array(prediction))**2
            soma += pa
        
        return soma

    def train(self, epochs, data_inputs, data_outputs, learning_rate, function_activation):
        epochs_total = epochs
        while epochs > 0:
            epochs-=1
            if(epochs % 10000 == 0):
                print(-epochs/epochs_total *100 + 100)
                print("MSE(erro medio quadratico) = ",self.MSE(data_inputs, data_outputs, function_activation))
            sample = rd.randint(0, len(data_inputs) - 1)
            data_input = np.array([data_inputs[sample]])
            data_output = np.array([data_outputs[sample]])
            predictions = self.predict(data_input, function_activation)
            predictions_error = self.calculateError(predictions, data_output)
            gradients = self.calculateGradient(predictions_error, predictions, learning_rate, function_activation)
            deltaWeights = self.calculateDeltaWeights(predictions, gradients, data_input)
            for i in range(len(self.weights)):
                self.weights[i] += deltaWeights[i]
                self.bias[i] += gradients[i]

    def predictAll(self, data_inputs, function_activation):
        predictions = []
        for i in range(len(data_inputs)):
            prediction = self.predict(np.array(data_inputs[i]), function_activation)[-1]
            predictions.append(np.array(prediction))
            print(data_inputs[i],"=",round(prediction[0][0]))
        return predictions
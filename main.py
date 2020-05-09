import numpy as np
import random as rd
import neuralnetwork as nn

def ReLU(x, derivative = False):
    if(derivative):
        return 1. * (x > 0)
    return x * (x > 0)

def sigmoid (x, derivative = False):
    if(derivative):
        return x * (1 - x)
    return 1/(1 + np.exp(-x))

data_outputs = np.array([[0],
                   [1],
                   [2],
                   [3],
                   [4],
                   [5],
                   [6],
                   [7],
                   [8],
                   [9]])
data_inputs = np.array([[0,0,0,0],
                            [0,0,0,1],
                            [0,0,1,0],
                            [0,0,1,1],
                            [0,1,0,0],
                            [0,1,0,1],
                            [0,1,1,0],
                            [0,1,1,1],
                            [1,0,0,0],
                            [1,0,0,1]])

#data_inputs = [[0,0],
#                [0,1],
#                [1,0],
#                [1,1]]
#data_outputs = [[0],
#                [1],
#                [1],
#                [0]]
#data_inputs = []
#data_outputs = []
#for i in range(1000):
#    data_input = [rd.randint(0,10), rd.randint(0,10)]
#    data_output = [data_input[0] + data_input[1]]
#    data_inputs.append(data_input)
#    data_outputs.append(data_output)

print("Olá, sou uma Rede Neural Artificial.\nMeu objetivo é aprender a converter numeros binarios em decimais")
np.set_printoptions(suppress=True)
epochs = 60000
learning_rate = 0.001
function_activation = ReLU
rede_neural = nn.NeuralNetwork([4,8,8,1])
rede_neural.predictAll(data_inputs, function_activation)
print("Essas são minhas predições sem passar por qualquer processo de apredizagem.")
print("Agora, baseada na biologia dos seres vivos, irei aprender com os meus erros!")
print("\nTreinar Rede Neural?")
input("")
while(rede_neural.MSE(data_inputs, data_outputs, function_activation) > 0.004):
    rede_neural.train(epochs, data_inputs, data_outputs, learning_rate, function_activation)
rede_neural.predictAll(data_inputs, function_activation)
print("Chupa otario!")
import random
import math
import numpy as np
import warnings


class NeuralNetwork:
    def __init__(self, layers=[1,1,1], thetas=None):
        self.network = [] #list of layers(which are lists of weight vectors (which are lists)) 
        self.activations = []
        self.layers = layers #list of values that represent how many neurons in each layer
        
        if thetas != None:
            self.network = thetas
            return
    
        
        for i in range(len(layers) - 1):
            layerWeights = np.random.uniform(-1, 1, size=(layers[i] + 1, layers[i + 1]))  # add 1 for bias node
            self.network.append(layerWeights)
        
        
    def safeExp(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.exp(x)

    
    def sigmoidTransfer(self, act):
        a = self.safeExp(-act)
        if a == '-inf' or a == 'inf':
            return .00000001
        
        return 1 / (1 + a)
    
    def sigmoidDerivative(self, x):
        return x * (1 - x)
    
    
    def forwardPropagation(self, instance, pr):
        if pr:
            print(f"processing instance: {instance}")
        layer = instance
        self.activations.append(instance)
        
        for i in range(len(self.network)):
            layer.insert(0, 1) #add bias node
            layerWeights = self.network[i]
            
            #####check if weights are one dimensional and change format if so####
            oneD = True
            for neuronWeights in layerWeights:
                if len(neuronWeights) != 1:
                    oneD = False
                    layerWeights = np.array(layerWeights).T
                    break
                
            if oneD:
                newLayerWeights = [i[0] for i in layerWeights]
                layerWeights = newLayerWeights
            ##############################################
                
            if pr:
                print(f"a{i + 1}: {layer} \n")
            #print(f"layer {i}'s weights: {layerWeights}")
            z = np.dot(layerWeights, layer)
            if pr:
                print(f"z{i + 2}: {z}")
            if np.isscalar(z):
                nextLayer = [self.sigmoidTransfer(float(z))]

            else:
                nextLayer = [self.sigmoidTransfer(activation) for activation in z]
                
            layer = nextLayer
            self.activations.append(layer)
            
            #print(f"activation (new neurons) for layer{i + 1}: {layer} \n")
            
        if pr:
            print(f"final activation: {layer} \n")
            print(f"f(x): {layer}")
        return layer
    
                
    
    def cost(self, data, values, l): #compute error 
        J = 0
        for instance, expected in zip(data, values):
            outputs = self.forwardPropagation(instance)
            if len(outputs) != len(expected):
                print("wrong size outputs")
                
            outputs = np.array(outputs)
            expected = np.array(expected)
            termA = np.multiply(-1 * expected, np.log(outputs))
            termB = np.multiply((-1 * expected) + 1, np.log((-1 * outputs) + 1))
            
            costs = np.subtract(termA, termB)
            sumCosts = sum(costs)
            print(f"Cost, J, associated with instance {instance}: {sumCosts}")
            J += sumCosts
            
        J /= len(data)
        print(f"final J value: {J}")
        
        
        S = 0
        for layer in self.network:
            for neuron in layer:
                for weight in neuron:
                    S += weight ** 2
                
        S *= (l / (2*len(data)))
        #print(f"final S value: {S}")
        #print(f"final regularized cost: {J + S}")
        
        return J + S
    
    
    def updateCost(self, J, n, l):
        #####cost calculations#####
        J /= n
        #print(f"final J value: {J}")
        S = 0
        for layer in self.network:
            for neuron in layer:
                for weight in neuron:
                    S += weight ** 2
                
        S *= (l / (2*n))
        #print(f"final S value: {S}")
        #print(f"final regularized cost: {J + S}")
        return J + S

    
    
    def backwardPropagation(self, data, values, l, alpha, epochs, pr):
        iterations = 0
        iterationsVsCost = {}
        
        for e in range(epochs):
            if pr:
                print(f"epoch {e}: ")
            gradients = []
            J = 0
            for instance, expected in zip(data, values):
                #####forward pass step#####
                i = instance.copy()
                self.activations = []
                outputs = np.array(self.forwardPropagation(i, pr))
                expected = np.array(expected)
                ###########################
                
                #####cost calculations#####
                termA = np.multiply(-1 * expected, np.log(outputs))
                termB = np.multiply((-1 * expected) + 1, np.log((-1 * outputs) + 1))
                
                costs = np.subtract(termA, termB)
                sumCosts = sum(costs)
                if pr:
                    print(f"Cost, J, associated with instance: {instance}: {sumCosts}")
                J += sumCosts
                iterations += 1
                iterationsVsCost[iterations] = sumCosts
                ###########################
                
                
                #get the delta of the output neurons
                outputDelta = np.subtract(outputs, expected)
                deltas = [outputDelta]
                if pr:
                    print(f"deltas for final layer: {deltas}")
                
                #calculate the deltas for the hidden layers
                for i in reversed(range(1, len(self.network))):
                    layerWeights = self.network[i]
                    
                    #######check if weights are one dimensional and change format if so######
                    oneD = True
                    for neuronWeights in layerWeights:
                        if len(neuronWeights) != 1:
                            oneD = False
                            layerWeights = np.array(layerWeights)
                            break
                        
                    if oneD:
                        newLayerWeights = [i[0] for i in layerWeights]
                        layerWeights = np.array([newLayerWeights]).T
                    ###################################

                    
                    
                    termA = np.dot(layerWeights, deltas[0])
                    termB = np.array(self.activations[i])
                    termC = 1 - termB
                    
                    deltaA = np.multiply(termA, termB)
                    delta = np.multiply(deltaA, termC)
                    
                    #delete the bias node
                    delta = np.delete(delta, 0)
                    
                    deltas.insert(0, delta)
                    if pr:
                        print(f"delta{i + 1}: {deltas[0]}")
                
                #compute the gradients from the deltas
                D = []  
                for i in reversed(range(len(self.network))):
                    termA = np.array([deltas[i]]).T
                    termB = np.array([self.activations[i]])
                    D.insert(0, np.dot(termA, termB))
                    
                for i in range(len(D)):
                    if pr:
                        print(f"gradients of theta{i+1} based on instance {instance}: {D[i]}")
                if pr:
                    print("\n\n\n")
                
                gradients.insert(0, D)
                
                
            #####cost calculations#####
            J /= len(data)
            #print(f"final J value: {J}")
            S = 0
            for layer in self.network:
                for i in range(1, len(layer)):
                    neuron = layer[i]
                    for weight in neuron:
                        S += weight ** 2
                    
            S *= (l / (2*len(data)))
            #print(f"final S value: {S}")
            finalCost = J + S
            if pr:
                print(f"Final (regularized) cost, J, based on the complete training set: {finalCost}")
            
            ############################

                
            #print(f"gradients: {gradients}")
            
            #sum the gradients of each instance 
            finalGradients = []
            for layerGradients in zip(*gradients):  
                layerSum = np.sum(layerGradients, axis=0)
                finalGradients.append(layerSum)
            #print(f"summed gradients: {finalGradients}")
            
            #regularize
            for i in reversed(range(len(self.network))):
                P = l * self.network[i]
                for j in range(len(P[0])):
                    P[0][j] = 0
                    
                P = P.T
                finalGradients[i] = np.add(finalGradients[i], P) / len(data)
                    
            if pr:
                for i in range(len(finalGradients)):
                    print(f"final regularized gradients of theta{i+1}: {finalGradients[i]}\n")
            
                    
            #update the weights
            for i in reversed(range(len(self.network))):
                termA = self.network[i]
                termB = alpha * finalGradients[i].T
                self.network[i] = np.subtract(termA, termB)
                
            #print(f"new network weights: {self.network}") 
            
        return iterationsVsCost
            
            
    def binaryClassification(self, test):
        classifications = []
        for instance in test:
            i = instance.copy()
            output = self.forwardPropagation(i, False)
            if output[0] > output[1]:
                classifications.append([1, 0])
                
            else:
                classifications.append([0, 1])
            
            
        return classifications
    
    
    def wineClassification(self, test):
        classifications = []
        for instance in test:
            i = instance.copy()
            output = self.forwardPropagation(i, False)
            if output[0] == max(output):
                classifications.append([1, 0, 0])
            elif output[1] == max(output):
                classifications.append([0, 1, 0])
                
            else:
                classifications.append([0, 0, 1])            
            
        return classifications
        
        
        
                
            
            
                
            
                
                
                
                
                

                
                    
                    
                
        
            
            
        
        
        
                
            
        
        
        
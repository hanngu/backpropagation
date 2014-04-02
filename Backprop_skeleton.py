import math
import random
import copy

#The transfer function of neurons, g(x)
def logFunc(x):
    return (1.0/(1.0+math.exp(-x)))

#The derivative of the transfer function, g'(x)
def logFuncDerivative(x):
    return math.exp(-x)/(pow(math.exp(-x)+1,2))

def randomFloat(low,high):
    return random.random()*(high-low) + low

#Initializes a matrix of all zeros
def makeMatrix(I, J):
    m = []
    for i in range(I):
        m.append([0]*J)
    return m

class NeuralNetwork:
    def __init__(self, numInputs, numHidden, learningRate=0.001):
        #Inputs: number of input and hidden nodes. Assuming a single output node.
        # +1 for bias node: A node with a constant input of 1. Used to shift the transfer function.
        self.numInputs = numInputs + 1
        self.numHidden = numHidden

        # Current activation levels for nodes (in other words, the nodes' output value)
        self.inputActivation = [1.0]*self.numInputs #B
        self.hiddenActivations = [1.0]*self.numHidden #B
        self.outputActivation = 1.0 #Assuming a single output. #O(b)
        self.learningRate = learningRate

        # create weights
        #A matrix with all weights from input layer to hidden layer
        self.weightsInput = makeMatrix(self.numInputs,self.numHidden)
        #A list with all weights from hidden layer to the single output neuron.
        self.weightsOutput = [0 for i in range(self.numHidden)]# Assuming single output
        # set them to random vaules
        for i in range(self.numInputs):
            for j in range(self.numHidden):
                self.weightsInput[i][j] = randomFloat(-0.5, 0.5)
        for j in range(self.numHidden):
            self.weightsOutput[j] = randomFloat(-0.5, 0.5)

        #Data for the backpropagation step in RankNets.
        #For storing the previous activation levels (output levels) of all neurons
        self.prevInputActivations = []
        self.prevHiddenActivations = []
        self.prevOutputActivation = 0 #O(a)
        #For storing the previous delta in the output and hidden layer
        self.prevDeltaOutput = 0
        self.prevDeltaHidden = [0 for i in range(self.numHidden)]
        #For storing the current delta in the same layers
        self.deltaOutput = 0
        self.deltaHidden = [0 for i in range(self.numHidden)]

    def propagate(self, inputs):
        if len(inputs) != self.numInputs-1:
            raise ValueError('wrong number of inputs')

        # input activations
        self.prevInputActivations=copy.deepcopy(self.inputActivation)
        for i in range(self.numInputs-1):
            self.inputActivation[i] = inputs[i]
        self.inputActivation[-1] = 1 #Set bias node to -1.

        # hidden activations
        self.prevHiddenActivations=copy.deepcopy(self.hiddenActivations)
        for j in range(self.numHidden):
            sum = 0.0
            for i in range(self.numInputs):
                #print self.ai[i] ," * " , self.wi[i][j]
                sum = sum + self.inputActivation[i] * self.weightsInput[i][j]
            self.hiddenActivations[j] = logFunc(sum)

        # output activations
        self.prevOutputActivation=self.outputActivation
        sum = 0.0
        for j in range(self.numHidden):
            sum = sum + self.hiddenActivations[j] * self.weightsOutput[j]
        self.outputActivation = logFunc(sum)
        return self.outputActivation

    def computeOutputDelta(self):
        #TODO: Implement the delta function for the output layer (see exercise text)
        pab =1.0/(1+math.exp(-(self.prevOutputActivation-self.outputActivation)))
        self.prevDeltaOutput = logFuncDerivative(self.prevOutputActivation) * (1.0-pab)
        self.deltaOutput = logFuncDerivative(self.outputActivation) * (1.0-pab)


    def computeHiddenDelta(self):
        #TODO: Implement the delta function for the hidden layer (see exercise text)
        for i in range(self.numHidden):
            self.prevDeltaHidden[i] = (logFuncDerivative(self.prevHiddenActivations[i])*self.weightsOutput[i]*(self.prevDeltaOutput - self.deltaOutput))
        for j in range(self.numHidden):
            self.deltaHidden[j] = (logFuncDerivative(self.hiddenActivations[j])*self.weightsOutput[j]*(self.prevDeltaOutput-self.deltaOutput))

    def updateWeights(self):
        #TODO: Update the weights of the network using the deltas (see exercise text)
        for i in range(self.numInputs):
            for j in range(self.numHidden):
                self.weightsInput[i][j] = self.weightsInput[i][j] + self.learningRate * (self.prevDeltaHidden[j] * self.prevInputActivations[i] - self.deltaHidden[j]*self.inputActivation[i])
        for i in range(self.numHidden):
            self.weightsOutput[i] = self.weightsOutput[i] + self.learningRate * (self.prevHiddenActivations[i] * self.prevDeltaOutput - self.hiddenActivations[i]* self.deltaOutput)

    def backpropagate(self):
        self.computeOutputDelta()
        self.computeHiddenDelta()
        self.updateWeights()

    #Prints the network weights
    def weights(self):
        print('Input weights:')
        for i in range(self.numInputs):
            print(self.weightsInput[i])
        print()
        print('Output weights:')
        print(self.weightsOutput)

    def train(self, patterns, iterations=1):
        #TODO: Train the network on all patterns for a number of iterations.
        #To measure performance each iteration: Run for 1 iteration, then count misordered pairs.
        #TODO: Training is done  like this (details in exercise text):
        for pattern in patterns:
            for pairs in pattern:
                a_propagated = self.propagate(pairs[0])
                b_propagated = self.propagate(pairs[1])
                self.backpropagate()

    def countMisorderedPairs(self, patterns):
        misses = 0
        correct = 0
        for pattern in patterns:
            for pairs in pattern:
                A = pairs[0]
                a_propagated = self.propagate(A)
                B = pairs[1]
                b_propagated = self.propagate(B)

                if a_propagated > b_propagated:
                    correct += 1
                elif a_propagated < b_propagated:
                    misses += 1

        return (0.0 + misses/(0.0 + correct + misses))

        #TODO: Let the network classify all pairs of patterns. The highest output determines the winner.
        #for each pair, do
        #Propagate A
        #Propagate B
        #if A>B: A wins. If B>A: B wins
        #if rating(winner) > rating(loser): numRight++
        #else: numMisses++
        #end of for
        #TODO: Calculate the ratio of correct answers:
        #errorRate = numMisses/(numRight+numMisses)
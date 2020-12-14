from perceptron_rosenblatt import *

# Start Perceptron with LR = 0,1 (learning rate) and epochs = 9500

def main():
    perceptron = PerceptronRosenblatt(learningRate=0.1, interations=9500)
    perceptron.calculateWeights()
    perceptron.plotErrors()

main()
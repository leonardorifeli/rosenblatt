import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PerceptronRosenblatt:

    def __init__(self, learningRate = 0.1, interations = 10):
        self.learningRate = learningRate
        self.errors = []
        self.interations = interations

        # LOAD CSV DATASET TO TRAINING
        dataFrame = pd.read_csv('training.data')
        y = dataFrame.iloc[0:60, 7].values

        # MARK WITH -1 ANY LESS MY RUs
        self.y = np.where(y=='no', -1, 1)

        self.X = dataFrame.iloc[0:60, 0:7].values

        # Generate first weights with zeros + first BIAS
        self.w = np.zeros(self.X.shape[1]+1)

    def calculateWeights(self):
        for _ in range(self.interations):
            errorCount = 0
            for xi, target in zip(self.X, self.y):
                predictedValue = self.predict(xi)
                wDelta = self.updateWeights(target, predictedValue, xi)
                errorCount += int(wDelta != 0.0)
            self.errors.append(errorCount)

        print("DONE! WEIGHTS:")
        for wFinal in zip(self.w):
            print(wFinal)

    def predict(self, xi):
        # SUM (x1 * w1, x2 * w2, xn * wn) + CURRENT_BIAS
        netInput = np.dot(xi, self.w[1:])+self.w[0]
        predictedValue = np.where(netInput > 0.0, 1, -1)        
        return predictedValue

    def updateWeights(self, target, predictedValue, xi):
        # Calculate wDelta (LEARNING_RATE * (TARGET-PREDICTED_VALUE)) * VALUES_INPUT
        wDelta = self.learningRate * (target - predictedValue)    
        self.w[1:] += wDelta * xi

        # Update BIAS
        self.w[0] += wDelta

        return wDelta

    def plotErrors(self):
        # Plot epochs and all misclassifications
        plt.plot(range(1, len(self.errors)+1), self.errors, marker='o', color='black')
        plt.xlabel('epochs')
        plt.ylabel('misclassifications')
        plt.show()


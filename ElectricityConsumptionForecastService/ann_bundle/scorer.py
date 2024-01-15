import math
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
class Scorer:
    def get_score(self, trainY, trainPredict, testY, testPredict):
        # trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
        # testScore = math.sqrt(mean_squared_error(testY, testPredict))
        trainScore = np.mean(np.abs((trainY - trainPredict)/trainY))*100
        testScore = np.mean(np.abs((testY - testPredict)/testY))*100
        return trainScore, testScore
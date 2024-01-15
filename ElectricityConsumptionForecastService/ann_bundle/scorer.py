import math
from sklearn.metrics import mean_squared_error
class Scorer:
    def get_score(self, trainY, trainPredict, testY, testPredict):
        trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
        testScore = math.sqrt(mean_squared_error(testY, testPredict))
        return trainScore, testScore
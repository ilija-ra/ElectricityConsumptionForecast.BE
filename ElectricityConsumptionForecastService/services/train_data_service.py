import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from ElectricityConsumptionForecast.utils.message_response import MessageResponse
from ElectricityConsumptionForecastRepository.repositories.train_data_repository import TrainRepository
from ElectricityConsumptionForecastRepository.repositories.weather_data_repository import WeatherRepository
from ElectricityConsumptionForecastService.ann_bundle.ann_regression import AnnRegression
from ElectricityConsumptionForecastService.ann_bundle.custom_preparer import CustomPreparer

# NUMBER_OF_COLUMNS = 20
SHARE_FOR_TRAINING = 0.85

class TrainService:
    def __init__(self, repository=None):
        self.train_repository = repository or TrainRepository()
        self.preprocessed_repository = repository or WeatherRepository()

    def train_with_regression(self, start_date, end_date):
        try:
            data = self.preprocessed_repository.get_all()

            if data.empty:
                return MessageResponse(success=False,message="Could not receive preprocessed data from a database").to_json()
            
            data = data[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)]
            
            data.drop('datetime', axis = 1, inplace= True)

            x_train_std, x_test_std, y_train, y_test = prepare_training_data_regression(data)

            poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
            x_inter_train = poly.fit_transform(x_train_std)
            x_inter_test = poly.transform(x_test_std)

            ridge_model = Ridge(alpha=1)
            ridge_model.fit(x_inter_train, y_train)

            return self.train_repository.save_regression_model(ridge_model)
        except Exception as e:
            return MessageResponse(success=False,message="Error occured while training regression model",errors=str(e)).to_json()

    def train_with_neural_network(self, start_date, end_date):
        try:
            data = self.preprocessed_repository.get_all()

            if data.empty:
                return MessageResponse(success=False,message="Could not receive preprocessed data from a database").to_json()
            
            data = data[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)]
            data.drop('datetime', axis = 1, inplace= True)

            preparer = CustomPreparer(data, len(data.columns), SHARE_FOR_TRAINING)
            trainX, trainY, testX, testY = preparer.prepare_for_training()

            ann_regression = AnnRegression()
            time_begin = time.time()
            ann_regression.compile_and_fit(trainX, trainY)
            time_end = time.time()

            return MessageResponse(success=True,message=f"Neural network trained successfully: Duration: {time_end - time_begin} seconds").to_json()
        except Exception as e:
            return MessageResponse(success=False,message="Error occured while training neural network model",errors=str(e)).to_json()

def prepare_training_data_regression(data: pd.DataFrame):
    x = data.drop('Load', axis=1)
    y = data['Load']
    y.bfill(inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_std = scaler.transform(x_train)
    x_test_std = scaler.transform(x_test)
    x_train_std = pd.DataFrame(x_train_std)
    x_test_std = pd.DataFrame(x_test_std)
    x_train_std.columns = list(x.columns)
    x_test_std.columns = list(x.columns)

    return x_train_std, x_test_std, y_train, y_test
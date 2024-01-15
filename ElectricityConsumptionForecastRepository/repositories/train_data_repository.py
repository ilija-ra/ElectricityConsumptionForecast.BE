import os
import pickle
import pathlib
from sklearn.linear_model import Ridge
from ElectricityConsumptionForecast.utils.message_response import MessageResponse

class TrainRepository:
    def save_regression_model(self, regression_model : Ridge):
        try:
            file_path = file_path = os.path.join(pathlib.Path().resolve(), 'ElectricityConsumptionForecastRepository', 'training_models', 'regression', 'linear_regression_ridge.pkl')
            with open(file_path, 'wb') as file:
                pickle.dump(regression_model, file)
            return MessageResponse(success=True,message="Regression model is successfully saved to a file linear_regression_ridge.pkl").to_json()
        except Exception as e:
            return MessageResponse(success=False,message="Error occured while saving regression model",errors=str(e)).to_json()

    def get_regression_model() -> Ridge:
        try:
            file_path = os.path.join(pathlib.Path().resolve(), 'ElectricityConsumptionForecastRepository', 'training_models', 'regression', 'linear_regression_ridge.pkl')
            with open(file_path, 'rb') as file:
                model = pickle.load(file)
            return model
        except:
            return

    def get_primary_regression_model() -> Ridge:
        try:
            file_path = os.path.join(pathlib.Path().resolve(), 'ElectricityConsumptionForecastRepository', 'training_models', 'regression', 'primary_linear_regression_ridge.pkl')
            with open(file_path, 'rb') as file:
                model = pickle.load(file)
            return model
        except:
            return
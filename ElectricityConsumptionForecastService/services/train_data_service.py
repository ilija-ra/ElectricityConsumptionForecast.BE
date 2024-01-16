import time
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import SGD
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from ElectricityConsumptionForecast.utils.message_response import MessageResponse
from ElectricityConsumptionForecastRepository.repositories.train_data_repository import TrainRepository
from ElectricityConsumptionForecastRepository.repositories.weather_data_repository import WeatherRepository
from ElectricityConsumptionForecastService.ann_bundle.ann_regression import AnnRegression

# NUMBER_OF_COLUMNS = 20
SHARE_FOR_TRAINING = 0.85
MODEL_NAME = 'current_model'
FILE_PATH = f"ElectricityConsumptionForecastRepository/training_models/neural_network/{MODEL_NAME}.keras"

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

            data['Load'] = data['Load'].astype(float)
            data['Load'].interpolate(inplace = True)
            
            y_train = data["Load"]
            x_train = data.drop("Load", axis=1)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(x_train)
            # X_test_scaled = scaler.transform(X_test)
            
            time_begin = time.time()

            model = keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
                layers.Dense(64, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(16, activation='relu'),
                layers.Dense(1)
            ])
            
            model.compile(optimizer='SGD', loss='mean_absolute_percentage_error')
            model.fit(X_train_scaled, y_train, epochs=150, batch_size=10, validation_split=0.2)
            
            time_end = time.time()
            
            ann_regression = AnnRegression()
            ann_regression.save_model(model, FILE_PATH)

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

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = tf.squeeze(y_true), tf.squeeze(y_pred)
    return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true)) * 100
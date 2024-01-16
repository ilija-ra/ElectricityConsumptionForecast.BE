import numpy as np
import pandas as pd
import tensorflow as tf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from ElectricityConsumptionForecast.utils.message_response import MessageResponse
from ElectricityConsumptionForecastRepository.repositories.train_data_repository import TrainRepository
from ElectricityConsumptionForecastRepository.repositories.weather_data_repository import WeatherRepository
from ElectricityConsumptionForecastRepository.repositories.a_weather_data_repository import AWeatherRepository
from ElectricityConsumptionForecastRepository.repositories.predict_values_repository import PredictRepository
from ElectricityConsumptionForecastService.ann_bundle.ann_regression import AnnRegression

SHARE_FOR_TRAINING = 0.85
MODEL_NAME = 'current_model'
FILE_PATH = f"ElectricityConsumptionForecastRepository/training_models/neural_network/{MODEL_NAME}.keras"
PRIMARY_FILE_PATH = f"ElectricityConsumptionForecastRepository/training_models/neural_network/primary_neural_network.keras"

class PredictService:
    def __init__(self, repository=None, service=None):
        self.preprocessed_repository = repository or WeatherRepository()
        self.a_preprocessed_repository = repository or AWeatherRepository()
        self.predict_repository = repository or PredictRepository()

    def get_prediction_result(self):
        return MessageResponse(success=True,message="Prediction result data",data=self.predict_repository.get_prediction_result()).to_json()

    def predict_with_regression(self, start_date, number_of_days):
        try:
            data = self.preprocessed_repository.get_all()

            if data.empty:
                return MessageResponse(success=False,message="Could not receive preprocessed data from a database").to_json()
            
            regression_model: Ridge = TrainRepository.get_regression_model()

            if regression_model is None:
                return MessageResponse(success=False,message="Could not receive regression model from a file").to_json()

            start_date = pd.to_datetime(start_date)
            end_date = start_date + pd.Timedelta(days=number_of_days)
            data = data[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)]
            
            data.to_csv("checking.csv")

            # Extract the 'datetime' column from the filtered data
            # date_time_column = data['datetime']
            # data.drop('datetime', axis = 1, inplace= True)

            x_inter_test, y_test = preparing_predict_data_regression(data)

            # Testiranje
            # y_predicted = regression_model.predict(x_inter_test)
            # prediction_mape = model_evaluation(y_test, y_predicted)

            # results=pd.concat([pd.DataFrame(x_test_datetime.values), pd.DataFrame(y_predicted)], axis=1)
            # results.columns = ['datetime', 'predicted_load']

            # csv_result = self.predict_repository.save_results_to_csv(results)
            # db_result = self.predict_repository.save_results_to_db(results)

            # return MessageResponse(success=True,message=f"{csv_result['message']} {db_result['message']}" ,result=prediction_mape).to_json()
        
        except Exception as e:
            return MessageResponse(success=False,message="Failed to predict data with regression", errors=str(e)).to_json()

    def predict_with_neural_network(self, start_date, number_of_days):
        try:
            data = self.preprocessed_repository.get_all()

            if data.empty:
                return MessageResponse(success=False, message="Could not receive preprocessed data from a database").to_json()

            start_date = pd.to_datetime(start_date)
            end_date = start_date + pd.Timedelta(days=number_of_days)

            data = data[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)]
            datetime_col = data['datetime']
            
            data.drop('datetime', axis=1, inplace=True)
            data['Load'] = data['Load'].astype(float)
            data['Load'].interpolate(inplace = True)

            y_test = data['Load']
            x_test = data.drop('Load', axis = 1)

            scaler = StandardScaler()
            X_test_scaled = scaler.fit_transform(x_test)
            # X_test_scaled = scaler.transform(X_train_scaled)

            ann_regression = AnnRegression()
            model = ann_regression.get_model_from_path(FILE_PATH)
            y_predicted = model.predict(X_test_scaled)

            y_test = tf.cast(y_test, dtype=tf.float32)
            y_predicted = tf.cast(y_predicted, dtype=tf.float32)

            mape_value = mean_absolute_percentage_error(y_test, y_predicted)
            
            print(mape_value)

            result_df = pd.concat([datetime_col.reset_index(drop=True), pd.Series(np.ravel(y_predicted), name='predicted_load')], axis=1)

            self.predict_repository.save_results_to_db(result_df)
            self.predict_repository.save_results_to_csv(result_df)

            print(mape_value)

            return MessageResponse(success=True, message="Prediction Successfully executed").to_json()
        except Exception as e:
            return MessageResponse(success=False, message="Failed to predict data with neural network", errors=str(e)).to_json()

    def predict_with_primary_neural_network(self):
        try:
            data = self.a_preprocessed_repository.a_get_all()
            
            if data.empty:
                return MessageResponse(success=False, message="Could not receive preprocessed data from a database").to_json()

            datetime_col = data['datetime']
            data.drop('datetime', axis=1, inplace=True)

            scaler = StandardScaler()
            X_test_scaled = scaler.fit_transform(data)
            # X_test_scaled = scaler.transform(X_train_scaled)

            ann_regression = AnnRegression()
            model = ann_regression.get_model_from_path(PRIMARY_FILE_PATH)
            y_predicted = model.predict(X_test_scaled)

            y_predicted = tf.cast(y_predicted, dtype=tf.float32)

            result_df = pd.concat([pd.Series(np.ravel(datetime_col), name='datetime'), 
                                   pd.Series(np.ravel(y_predicted), name='predicted_load')], axis=1)

            self.predict_repository.save_results_to_csv(result_df)
            self.predict_repository.save_results_to_db(result_df)

            return MessageResponse(success=True, message="Prediction Successfully executed").to_json()
        except Exception as e:
            return MessageResponse(success=False, message="Failed to predict data with neural network", errors=str(e)).to_json()

def preparing_predict_data_regression(data: pd.DataFrame):
    # U x smesteni prediktori, dok y varijablu treba predvideti
    x = data.drop('Load', axis=1)
    y = data['Load']
    y.bfill(inplace=True)

    # podela skupa na trening i test podatke
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

    # Selekcija obelezja
    X = sm.add_constant(x_train)
    model = sm.OLS(y_train, X.astype('float')).fit()
    model.summary()

    # Standardizacija obelezja (svodjenje na sr.vr. 0 i varijansu 1)
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_std = scaler.transform(x_train)
    x_test_std = scaler.transform(x_test)
    x_train_std = pd.DataFrame(x_train_std)
    x_test_std = pd.DataFrame(x_test_std)
    x_train_std.columns = list(x.columns)
    x_test_std.columns = list(x.columns)

    # ova hipoteza obuhvata i drugi stepen obelezja
    poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
    x_inter_train = poly.fit_transform(x_train_std)
    x_inter_test = poly.transform(x_test_std)

    return x_inter_test, y_test

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
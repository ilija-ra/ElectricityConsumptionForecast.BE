import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from ElectricityConsumptionForecast.utils.message_response import MessageResponse
from ElectricityConsumptionForecastRepository.repositories.train_data_repository import TrainRepository
from ElectricityConsumptionForecastRepository.repositories.weather_data_repository import WeatherRepository
from ElectricityConsumptionForecastRepository.repositories.predict_values_repository import PredictRepository
from ElectricityConsumptionForecastService.ann_bundle.ann_regression import AnnRegression
from ElectricityConsumptionForecastService.ann_bundle.custom_preparer import CustomPreparer
from ElectricityConsumptionForecastService.ann_bundle.scorer import Scorer

SHARE_FOR_TRAINING = 0.85

class PredictService:
    def __init__(self, repository=None, service=None):
        self.preprocessed_repository = repository or WeatherRepository()
        self.predict_repository = repository or PredictRepository()

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
            y_predicted = regression_model.predict(x_inter_test)
            prediction_mape = model_evaluation(y_test, y_predicted)

            results=pd.concat([pd.DataFrame(x_test_datetime.values), pd.DataFrame(y_predicted)], axis=1)
            results.columns = ['datetime', 'predicted_load']

            csv_result = self.predict_repository.save_results_to_csv(results)
            db_result = self.predict_repository.save_results_to_db(results)

            return MessageResponse(success=True,message=f"{csv_result['message']} {db_result['message']}" ,result=prediction_mape).to_json()
        
        except Exception as e:
            return MessageResponse(success=False,message="Failed to predict data with regression", errors=str(e)).to_json()

    def predict_with_neural_network(self, start_date, number_of_days):
        try:
            data = self.preprocessed_repository.get_all()

            if data.empty:
                return MessageResponse(success=False,message="Could not receive preprocessed data from a database").to_json()
            
            start_date = pd.to_datetime(start_date)
            end_date = start_date + pd.Timedelta(days=number_of_days)

            data = data[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)]
            data.drop('datetime', axis = 1, inplace= True)

            preparer = CustomPreparer(data, len(data.columns), SHARE_FOR_TRAINING)
            trainX, trainY, testX, testY = preparer.prepare_for_training()
            
            ann_regression = AnnRegression()
            time_begin = time.time()
            trainPredict, testPredict = ann_regression.compile_fit_predict(trainX, trainY, testX)
            time_end = time.time()

            # invert predictions
            trainPredict, trainY, testPredict, testY = preparer.inverse_transform(trainPredict, testPredict)

            # calculate root mean squared error
            scorer = Scorer()
            trainScore, testScore = scorer.get_score(trainY, trainPredict, testY, testPredict)
            print('Train Score: %.2f MAPE' % (trainScore))
            print('Test Score: %.2f MAPE' % (testScore))

            # np.savetxt("filesa.txt", trainX['day'], fmt='%d', delimiter='\t')

            return MessageResponse(success=True,message="Prediction Successfully executed", result=testScore).to_json()
        except Exception as e:
            return MessageResponse(success=False,message="Failed to predict data with neural network", errors=str(e)).to_json()

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

def model_evaluation(y, y_predicted):
    mape = mean_absolute_percentage_error(y, y_predicted)
    return (mape * 100).round(5)

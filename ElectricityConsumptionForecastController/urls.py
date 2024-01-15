from django.urls import path
from ElectricityConsumptionForecastController.controllers import load_data_controller, weather_data_controller, train_data_controller, predict_data_controller

urlpatterns = [
    path('load/get', load_data_controller.get_all, name='load-data-get-all'),
    path('load/get_head_of_data', load_data_controller.get_head_of_data, name='load-data-get-head'),
    # path('load/save/', load_data_controller.save, name='load-data-save'),
    path('weather/get', weather_data_controller.get_all, name='weather-data-get-all'),
    path('weather/get_head_of_data', weather_data_controller.get_head_of_data, name='weather-data-get-head'),
    path('weather/save/', weather_data_controller.save, name='weather-data-save'),
    path('weather/delete', weather_data_controller.delete_all, name='weather-data-delete-all'),
    # path('train/primary_regression', train_data_controller.train_with_primary_regression, name='train-with-primary-regression'),
    path('train/regression', train_data_controller.train_with_regression, name='train-with-regression'),
    path('predict/regression', predict_data_controller.predict_with_regression, name='predict-with-regression'),
    path('train/neural_network', train_data_controller.train_with_neural_network, name='train-with-neural-network'),
    path('predict/neural_network', predict_data_controller.predict_with_neural_network, name='predict-with-neural-network'),
    path('predict/get_result', predict_data_controller.get_prediction_result, name='get-predict-result')
]
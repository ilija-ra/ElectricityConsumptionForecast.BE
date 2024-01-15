from rest_framework import serializers
from ElectricityConsumptionForecastRepository.models.preprocess_data import ProcessedForecastData

class ProcessedForecastDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProcessedForecastData
        fields = ('id',
                  'temp',
                  'feelslike',
                  'dew',
                  'humidity',
                  'windspeed',
                  'winddir',
                  'sealevelpressure',
                  'cloudcover',
                  'visibility',
                  'uvindex',
                  'conditions',
                  'Time_Zone',
                  'Load',
                  'year',
                  'month',
                  'day',
                  'hour',
                  'day_of_week',
                  'Load_avg_prev_day',
                  'temp_avg_prev_day')
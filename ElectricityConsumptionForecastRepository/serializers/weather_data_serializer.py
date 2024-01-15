from rest_framework import serializers
from ElectricityConsumptionForecastRepository.models.weather_data import RawForecastWeatherData

class RawForecastWeatherDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = RawForecastWeatherData
        fields = ('id',
                  'name',
                  'datetime',
                  'temp',
                  'feelslike',
                  'dew',
                  'humidity',
                  'precip',
                  'precipprob',
                  'preciptype',
                  'snow',
                  'snowdepth',
                  'windgust',
                  'windspeed',
                  'winddir',
                  'sealevelpressure',
                  'cloudcover',
                  'visibility',
                  'solarradiation',
                  'solarenergy',
                  'uvindex',
                  'severerisk',
                  'conditions')
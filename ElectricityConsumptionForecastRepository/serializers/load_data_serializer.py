from rest_framework import serializers
from ElectricityConsumptionForecastRepository.models.load_data import RawForecastLoadData

class RawForecastLoadDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = RawForecastLoadData
        fields = ('id',
                  'TimeStamp',
                  'TimeZone',
                  'Name',
                  'PTID',
                  'Load')
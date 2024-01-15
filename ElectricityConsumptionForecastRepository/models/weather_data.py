from django.db import models

class RawForecastWeatherData(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255, null=True, blank=True)
    datetime = models.DateTimeField(null=True, blank=True)
    temp = models.FloatField(null=True, blank=True)
    feelslike = models.FloatField(null=True, blank=True)
    dew = models.FloatField(null=True, blank=True)
    humidity = models.FloatField(null=True, blank=True)
    precip = models.FloatField(null=True, blank=True)
    precipprob = models.FloatField(null=True, blank=True)
    preciptype = models.FloatField(null=True, blank=True)
    snow = models.FloatField(null=True, blank=True)
    snowdepth = models.FloatField(null=True, blank=True)
    windgust = models.FloatField(null=True, blank=True)
    windspeed = models.FloatField(null=True, blank=True)
    winddir = models.IntegerField(null=True, blank=True)
    sealevelpressure = models.FloatField(null=True, blank=True)
    cloudcover = models.FloatField(null=True, blank=True)
    visibility = models.FloatField(null=True, blank=True)
    solarradiation = models.IntegerField(null=True, blank=True)
    solarenergy = models.FloatField(null=True, blank=True)
    uvindex = models.IntegerField(null=True, blank=True)
    severerisk = models.BooleanField(null=True, blank=True)
    conditions = models.CharField(max_length=255, null=True, blank=True)
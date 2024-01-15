from django.db import models

class RawForecastLoadData(models.Model):
    id = models.AutoField(primary_key=True)
    TimeStamp = models.CharField(max_length=255, null=True, blank=True)
    TimeZone = models.CharField(max_length=255, null=True, blank=True)
    Name = models.CharField(max_length=255, null=True, blank=True)
    PTID = models.BigIntegerField(null=True, blank=True)
    Load = models.FloatField(null=True, blank=True)
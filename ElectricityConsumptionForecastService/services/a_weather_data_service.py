import numpy as np
import pandas as pd
from ElectricityConsumptionForecast.utils.message_response import MessageResponse
from ElectricityConsumptionForecastRepository.repositories.a_weather_data_repository import AWeatherRepository

class AWeatherService:
    def __init__(self, repository=None):
        self.weather_repository = repository or AWeatherRepository()
    
    def a_get_head_of_data(self):
        return MessageResponse(success=True,message="First 10 rows of preprocessed data",data=self.weather_repository.a_get_head_of_data()).to_json()

    def a_get_all(self):
        return MessageResponse(success=True,message="Getting preprocessed data was successful",data=self.weather_repository.a_get_all()).to_json()

    def a_save(self, data):
        merged_weather_data_df = a_merged_files_as_dataframe(data)
        preprocessed_data_df = a_preprocess_data(merged_weather_data_df)
        
        if preprocessed_data_df.empty:
            return MessageResponse(success=False,message="Error occured while executing preprocessing operation on data").to_json()

        return self.weather_repository.a_save(preprocessed_data_df)

    def a_delete_all(self):
        return self.weather_repository.a_delete_all()

def a_merged_files_as_dataframe(files: any) -> pd.DataFrame:
    merged_data = pd.DataFrame()
    for file in files:
        csv_data = pd.read_csv(file)
        merged_data = pd.concat([merged_data, csv_data], ignore_index=True)
    return merged_data

def a_preprocess_data(weather_df: pd.DataFrame) -> pd.DataFrame:
    try:
        # weather_df = weather_df['name', 'datetime', 'temp', 'feelslike', 'dew', 'humidity', 'precip', 'precipprob', 'preciptype', 'snow', 'snowdepth', 'windgust', 'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation', 'solarenergy', 'uvindex', 'severerisk', 'conditions']
        # # # # Weather type file preprocessing
        weather_df.drop(['name', 'preciptype', 'severerisk', 'precipprob', 'windgust', 'solarenergy', 'solarradiation', 'precip', 'snow', 'snowdepth', 'icon', 'stations'], axis = 1, inplace= True)
        weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
        weather_df['temp'] = weather_df['temp'].astype(float)
        weather_df.loc[weather_df['temp'] > 134, 'temp'] = np.nan

        weather_df['temp'].interpolate(inplace = True)
        weather_df['feelslike'].interpolate(inplace = True)
        weather_df['dew'].interpolate(inplace = True)
        weather_df['windspeed'].interpolate(inplace = True)
        weather_df['winddir'].interpolate(inplace = True)
        weather_df['sealevelpressure'].interpolate(inplace = True)
        weather_df['cloudcover'].interpolate(inplace = True)
        weather_df['visibility'].interpolate(inplace = True)

        weather_df.loc[weather_df['conditions'] == "Clear", 'conditions'] = 0
        weather_df.loc[weather_df['conditions'] == "Partially cloudy", 'conditions'] = 1
        weather_df.loc[weather_df['conditions'] == "Overcast", 'conditions'] = 2
        weather_df.loc[weather_df['conditions'] == "Rain", 'conditions'] = 3
        weather_df.loc[weather_df['conditions'] == "Snow", 'conditions'] = 4
        weather_df.loc[weather_df['conditions'] == "Rain, Overcast", 'conditions'] = 5
        weather_df.loc[weather_df['conditions'] == "Rain, Partially cloudy", 'conditions'] = 6
        weather_df.loc[weather_df['conditions'] == "Snow, Partially cloudy", 'conditions'] = 7
        weather_df.loc[weather_df['conditions'] == "Snow, Overcast", 'conditions'] = 8

        weather_df['humidity'].bfill(inplace = True)
        weather_df['conditions'].bfill(inplace = True)
        
        weather_df['year'] = weather_df['datetime'].dt.year
        weather_df['month'] = weather_df['datetime'].dt.month
        weather_df['day'] = weather_df['datetime'].dt.day
        weather_df['hour'] = weather_df['datetime'].dt.hour
        weather_df['day_of_week'] = weather_df['datetime'].dt.dayofweek + 1

        # Calculate the average temperature for each day
        daily_avg_temp = weather_df.groupby(weather_df['datetime'].dt.date)['temp'].mean()

        # Merge the daily average temperature back to the original DataFrame
        weather_df = weather_df.merge(daily_avg_temp, left_on=weather_df['datetime'].dt.date, right_index=True, suffixes=('', '_avg_prev_day'))
        weather_df.drop('key_0', axis = 1, inplace= True)

        weather_df = weather_df.round(2)

        return weather_df
    
    except Exception as e:
        return pd.DataFrame()
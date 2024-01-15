import numpy as np
import pandas as pd
from ElectricityConsumptionForecast.utils.message_response import MessageResponse
from ElectricityConsumptionForecastRepository.repositories.load_data_repository import LoadRepository
from ElectricityConsumptionForecastRepository.repositories.weather_data_repository import WeatherRepository

class WeatherService:
    def __init__(self, repository=None):
        self.weather_repository = repository or WeatherRepository()
        self.load_repository = repository or LoadRepository()
    
    def get_head_of_data(self):
        return MessageResponse(success=True,message="First 10 rows of preprocessed data",data=self.weather_repository.get_head_of_data()).to_json()

    def get_all(self):
        return MessageResponse(success=True,message="Getting preprocessed data was successful",data=self.weather_repository.get_all()).to_json()

    def save(self, data):
        
        merged_weather_data_df = merged_files_as_dataframe(data)
        preprocessed_data_df = preprocess_data(merged_weather_data_df, self.load_repository.get_all())

        if preprocessed_data_df.empty:
            return MessageResponse(success=False,message="Error occured while executing preprocessing operation on data").to_json()

        return self.weather_repository.save(preprocessed_data_df)

    def delete_all(self):
        return self.weather_repository.delete_all()

def merged_files_as_dataframe(files: any) -> pd.DataFrame:
    merged_data = pd.DataFrame()
    for file in files:
        csv_data = pd.read_csv(file)
        merged_data = pd.concat([merged_data, csv_data], ignore_index=True)
    return merged_data

def preprocess_data(weather_df: pd.DataFrame, load_df: pd.DataFrame) -> pd.DataFrame:
    try:
        # # # Load type file preprocessing
        load_df.rename(columns = {'TimeStamp':'datetime'}, inplace = True)
        load_df = load_df[load_df.Name == 'N.Y.C.']
        load_df['datetime'] = pd.to_datetime(load_df['datetime'])
        load_df = load_df[(load_df['datetime'].dt.minute == 0) & (load_df['datetime'].dt.second == 0)]
        load_df.drop(['id', 'Name', 'PTID'], axis = 1, inplace= True)
        load_df['Load'].interpolate(inplace = True)

        # # # Weather type file preprocessing
        weather_df.drop(['name', 'preciptype', 'severerisk', 'precipprob', 'windgust', 'solarenergy', 'solarradiation', 'precip', 'snow', 'snowdepth',], axis = 1, inplace= True)
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
        
        # # # Merged data preprocessing
        weather_load_df = pd.merge(weather_df, load_df, on='datetime')

        weather_load_df['year'] = weather_load_df['datetime'].dt.year
        weather_load_df['month'] = weather_load_df['datetime'].dt.month
        weather_load_df['day'] = weather_load_df['datetime'].dt.day
        weather_load_df['hour'] = weather_load_df['datetime'].dt.hour
        weather_load_df['day_of_week'] = weather_load_df['datetime'].dt.dayofweek + 1
        
        # Calculate the average consumption for each day
        weather_load_df['Load'] = weather_load_df['Load'].astype(float)
        daily_avg_load = weather_load_df.groupby(weather_load_df['datetime'].dt.date)['Load'].mean()
        
        # Merge the daily average consumption back to the original DataFrame
        weather_load_df = weather_load_df.merge(daily_avg_load, left_on=weather_load_df['datetime'].dt.date, right_index=True, suffixes=('', '_avg_prev_day'))
        weather_load_df.drop('key_0', axis = 1, inplace= True)
        
        # Calculate the average temperature for each day
        daily_avg_temp = weather_load_df.groupby(weather_load_df['datetime'].dt.date)['temp'].mean()

        # Merge the daily average temperature back to the original DataFrame
        weather_load_df = weather_load_df.merge(daily_avg_temp, left_on=weather_load_df['datetime'].dt.date, right_index=True, suffixes=('', '_avg_prev_day'))
        weather_load_df.drop('key_0', axis = 1, inplace= True)
        
        # weather_load_df.drop('datetime', axis = 1, inplace= True)

        weather_load_df.loc[weather_load_df['TimeZone'] == "EDT", 'TimeZone'] = 1
        weather_load_df.loc[weather_load_df['TimeZone'] == "EST", 'TimeZone'] = 2
        weather_load_df = weather_load_df.round(2)

        return weather_load_df
    
    except Exception as e:
        return pd.DataFrame()
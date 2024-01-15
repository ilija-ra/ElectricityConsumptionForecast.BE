import pandas as pd
from injector import inject
from ElectricityConsumptionForecast.utils.message_response import MessageResponse
from ElectricityConsumptionForecastRepository.repositories.load_data_repository import LoadRepository

class LoadService:
    @inject
    def __init__(self, repository=None):
        self.repository = repository or LoadRepository()

    def get_head_of_data(self):
        return MessageResponse(success=True,message="First 10 rows of load data",data=self.repository.get_head_of_data()).to_json()

    def get_all(self):
        return MessageResponse(success=True,message="Getting load data was successful",data=self.repository.get_all()).to_json()

    def save(self, data):
        merged_weather_data_df = merged_files_as_dataframe(data)
        return self.repository.save(merged_weather_data_df)
    
def merged_files_as_dataframe(files: any) -> pd.DataFrame:
    merged_data = pd.DataFrame()
    for file in files:
        csv_data = pd.read_csv(file)
        merged_data = pd.concat([merged_data, csv_data], ignore_index=True)
    return merged_data
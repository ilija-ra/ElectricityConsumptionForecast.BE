import sqlite3
import pandas as pd
from django.db import connections
from ElectricityConsumptionForecast.utils.message_response import MessageResponse

class WeatherRepository:
    def get_head_of_data(self):
        return pd.read_sql('''SELECT * FROM PreprocessedForecastData LIMIT 10''', connections['default'])

    def get_all(self) -> pd.DataFrame:
        return pd.read_sql('''SELECT * FROM PreprocessedForecastData''', connections['default'])

    def save(self, data : pd.DataFrame):
        try:
            conn = sqlite3.connect('db.sqlite3')
            c = conn.cursor()
            data.columns = data.columns.str.replace(' ', '')
            columns = ', '.join([f'{column} TEXT' for column in data.columns])
            c.execute(f'CREATE TABLE IF NOT EXISTS PreprocessedForecastData (id INTEGER PRIMARY KEY, {columns})')
            data.to_sql('PreprocessedForecastData', conn, if_exists='replace', index=False)
            conn.commit()
            conn.close()
            return MessageResponse(success=True,message="Saving preprocessed data was successful").to_json()
        except Exception as e:
            return MessageResponse(success=False,message="Error occured while saving preprocessed data",errors=str(e)).to_json()

    def delete_all(self):
        try:
            conn = sqlite3.connect('db.sqlite3')
            c = conn.cursor()
            c.execute('DELETE FROM PreprocessedForecastData')
            conn.commit()
            conn.close()
            return MessageResponse(success=True,message="Preprocessed data successfully deleted").to_json()
        except Exception as e:
            return MessageResponse(success=False,message="Error occured while deleting preprocessed data",errors=str(e)).to_json()
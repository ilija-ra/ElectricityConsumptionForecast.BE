import sqlite3
import pandas as pd
from django.db import connections
from ElectricityConsumptionForecast.utils.message_response import MessageResponse

class AWeatherRepository:
    def a_get_head_of_data(self):
        return pd.read_sql('''SELECT * FROM APreprocessedForecastData LIMIT 10''', connections['default'])

    def a_get_all(self) -> pd.DataFrame:
        return pd.read_sql('''SELECT * FROM APreprocessedForecastData''', connections['default'])

    def a_save(self, data : pd.DataFrame):
        try:
            conn = sqlite3.connect('db.sqlite3')
            c = conn.cursor()
            data.columns = data.columns.str.replace(' ', '')
            columns = ', '.join([f'{column} TEXT' for column in data.columns])
            c.execute(f'CREATE TABLE IF NOT EXISTS APreprocessedForecastData (id INTEGER PRIMARY KEY, {columns})')
            data.to_sql('APreprocessedForecastData', conn, if_exists='replace', index=False)
            conn.commit()
            conn.close()
            return MessageResponse(success=True,message="Saving preprocessed data was successful").to_json()
        except Exception as e:
            return MessageResponse(success=False,message="Error occured while saving preprocessed data",errors=str(e)).to_json()
        
    def a_delete_all(self):
        try:
            conn = sqlite3.connect('db.sqlite3')
            c = conn.cursor()
            c.execute('DELETE FROM APreprocessedForecastData')
            conn.commit()
            conn.close()
            return MessageResponse(success=True,message="Preprocessed data successfully deleted").to_json()
        except Exception as e:
            return MessageResponse(success=False,message="Error occured while deleting preprocessed data",errors=str(e)).to_json()
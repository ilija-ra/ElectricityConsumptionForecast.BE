import sqlite3
import pandas as pd
from django.db import connections
from ElectricityConsumptionForecast.utils.message_response import MessageResponse

class LoadRepository:
    def get_head_of_data(self):
        return pd.read_sql('''SELECT * FROM RawForecastLoadData LIMIT 10''', connections['default'])
    
    def get_all(self) -> pd.DataFrame:
        return pd.read_sql('''SELECT * FROM RawForecastLoadData''', connections['default'])
    
    def save(self, data : pd.DataFrame):
        try:
            conn = sqlite3.connect('db.sqlite3')
            c = conn.cursor()
            data.columns = data.columns.str.replace(' ', '')
            columns = ', '.join([f'{column} TEXT' for column in data.columns])
            c.execute(f'CREATE TABLE IF NOT EXISTS RawForecastLoadData (id INTEGER PRIMARY KEY, {columns})')
            data.to_sql('RawForecastLoadData', conn, if_exists='append', index=False)
            conn.commit()
            conn.close()
            return MessageResponse(success=True,message="Saving load data was successful").to_json()
        except Exception as e:
            return MessageResponse(success=False,message="Error occured while saving load data",errors=str(e)).to_json()
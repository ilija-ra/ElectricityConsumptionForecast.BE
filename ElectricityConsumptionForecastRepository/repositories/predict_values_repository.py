import sqlite3
from django.db import connections
import pandas as pd
from ElectricityConsumptionForecast.utils.message_response import MessageResponse

class PredictRepository:
    def get_prediction_result(self):
        return pd.read_sql('''SELECT * FROM PredictedForecastData''', connections['default'])

    def save_results_to_db(self, data : pd.DataFrame):
        try:
            conn = sqlite3.connect('db.sqlite3')
            c = conn.cursor()
            data.columns = data.columns.str.replace(' ', '')
            columns = ', '.join([f'{column} TEXT' for column in data.columns])
            c.execute(f'CREATE TABLE IF NOT EXISTS PredictedForecastData (id INTEGER PRIMARY KEY, {columns})')
            data.to_sql('PredictedForecastData', conn, if_exists='replace', index=False)
            conn.commit()
            conn.close()
            return MessageResponse(success=True,message="Predicted results successfully saved to database").to_json()
        except Exception as e:
            return MessageResponse(success=False,message="Error occured while saving results to a database",errors=str(e)).to_json()
    
    def save_results_to_csv(self, data_to_save: pd.DataFrame):
        try:
            data_to_save.to_csv("prediction_results.csv", index=False)
            return MessageResponse(success=True,message="Predicted results successfully saved to csv file").to_json()
        except Exception as e:
            return MessageResponse(success=False,message="Error occured while saving results to csv file",errors=str(e)).to_json()
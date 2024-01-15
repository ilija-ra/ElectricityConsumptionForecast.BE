import json
import pandas as pd

class MessageResponse:
    def __init__(self, success=True, message: str=None, errors: str=None, result=None, data: pd.DataFrame=None):
            self.success = success
            self.message = message
            self.errors = errors
            self.result = result
            
            if data is not None and not data.empty:
                self.data = json.loads(data.to_json(orient="index"))
            else:
                self.data = None

    def to_json(self):
        return json.loads(json.dumps(self.__dict__))
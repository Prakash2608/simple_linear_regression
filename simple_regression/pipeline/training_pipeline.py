import os
import sys
import pandas as pd
import numpy as np

from simple_regression.exception import AppException
from simple_regression.logger import logging
from simple_regression.utils.main_utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    
    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model = load_object(file_path = model_path)
            preprocessor = load_object(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            
            return preds
            
        except Exception as e:
            raise AppException(e,sys)
        
        
        
class CustomData:
    def __init__(self,
                 carwidth: float,
                 carheight: float,
                 fueltypes: str,
                 carbody: str):
        self.carwidth = carwidth
        self.carheight = carheight
        self.fueltypes = fueltypes
        self.carbody = carbody
        
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "carwidth":[self.carwidth],
                "carheight": [self.carheight],
                "fueltypes": [self.fueltypes],
                "carbody": [self.carbody]
            }
            
            return pd.DataFrame(custom_data_input_dict)
            
        except Exception as e:
            raise AppException(e,sys)
import os
import sys
import pandas as pd
from simple_regression.logger import logging
from simple_regression.exception import AppException

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from simple_regression.constant.training_pipeline import *
from simple_regression.entity.config_entity import (DataIngestionConfig)



    
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
        
    def initiate_data_ingetsion(self):
        logging.info("Entered the data ingestion method or component")
        
        try:
            df = pd.read_csv('data\car_dataset.csv')
            logging.info("Read the dataset as dataframe")
            df = df[['carwidth', 'carheight','fueltypes','carbody','price']]
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header = True)
            
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header= True)
            
            logging.info("ingestion of the data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path, 
                self.ingestion_config.raw_data_path
            )
            
        except Exception as e:
            raise AppException(e, sys)
        
        
# if __name__=="__main__":
#     obj = DataIngestion()
#     train_data, test_data= obj.initiate_data_ingetsion()
    
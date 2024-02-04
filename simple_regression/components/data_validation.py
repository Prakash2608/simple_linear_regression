import sys 
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from simple_regression.exception import AppException
from simple_regression.logger import logging
from simple_regression.utils.main_utils import save_object


@dataclass
class DataTransformationConfig:
    prepprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
        
    def get_data_transformation_object(self):
        """
        This function is responsible for data transformation
        """
        try:
            numerical_columns = ['carwidth', 'carheight']
            categorical_columns = ['fueltypes','carbody']
            
            num_pipeline = Pipeline(
                steps = [
                    ('scaler', StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps = [
                    ("onehotencoder", OneHotEncoder())
                ]
            )
            
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"categorical columns: {categorical_columns}")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise AppException(e,sys)
        

    def initiate_data_transformation(self, train_path, test_path, raw_path):
        try:
            # train_df = pd.read_csv(train_path)
            # test_df = pd.read_csv(test_path)
            raw_df = pd.read_csv(raw_path)
            
            logging.info("Read train and test data completed")
            
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformation_object()
            
            # independent_features = ['carwidth', 'carheight','fueltypes','carbody']
            # dependent_feature = ['price']
            
            input_feature_raw_df = raw_df[['carwidth', 'carheight','fueltypes','carbody']]
            target_feature_raw_df = raw_df['price']
            
            # input_feature_test_df = test_df[['carwidth', 'carheight','fueltypes','carbody']]
            # target_feature_test_df = test_df['price']
            
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )            
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_raw_df)
            # input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)
            
            print("input_feature_train_arr shape: ", len(input_feature_train_arr[0]))
            # print("input_feature_test_arr shape: " , len(input_feature_test_arr[0]))
            
            
            # X_train, 
            # separate dataset into train and test
            
            X_train, X_test, y_train, y_test = train_test_split(input_feature_train_arr,target_feature_raw_df,test_size=0.2,random_state=42)
            
            train_arr = np.c_[
                X_train, np.array(y_train)
            ]
            test_arr = np.c_[X_test, np.array(y_test)]
            
            logging.info(f"Saved preprocessing object.")
            
            save_object(
                file_path = self.data_transformation_config.prepprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.prepprocessor_obj_file_path
            )
                        
        except Exception as e:
            raise AppException(e,sys)     
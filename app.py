# from simple_regression.components.data_ingestion import DataIngestion
# from simple_regression.components.data_validation import DataTransformation
# from simple_regression.components.model_trainer import ModelTrainer


# if __name__=="__main__":
#     obj = DataIngestion()
#     train_data, test_data, raw_data= obj.initiate_data_ingetsion()
    
#     data_transformation = DataTransformation()
#     train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data, raw_data)
    
#     model_trainer = ModelTrainer()
#     print(model_trainer.initiate_model_trainer(train_arr, test_arr))
    
    

from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from simple_regression.exception import AppException
from simple_regression.logger import logging

from sklearn.preprocessing import StandardScaler
from simple_regression.pipeline.training_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predictdata', methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html') 
    
    else:
        data = CustomData(
            carwidth = request.form.get('carwidth'),
            carheight = request.form.get('carheight'),
            fueltypes = request.form.get('fueltypes'),
            carbody = request.form.get('carbody')
        )   
        
        pred_df = data.get_data_as_data_frame()
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        return render_template('home.html', results=results[0])
    
    
if __name__ == '__main__':
    app.run(debug=True)
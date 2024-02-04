import os
from dataclasses import dataclass
from datetime import datetime
from simple_regression.constant.training_pipeline import *



@dataclass
class TrainingPipelineConfig:
    artifacts_dir: str = ARTIFACTS_DIR



training_pipeline_config:TrainingPipelineConfig = TrainingPipelineConfig() 


@dataclass
class DataIngestionConfig:
    train_data_path: str= os.path.join(training_pipeline_config.artifacts_dir, 'train.csv')
    test_data_path: str= os.path.join(training_pipeline_config.artifacts_dir, 'test.csv')
    raw_data_path: str= os.path.join(training_pipeline_config.artifacts_dir, 'data.csv')



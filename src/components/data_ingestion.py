import os 
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from components.model_trainer import ModelTrainer, ModelTrainerConfig

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook/Data/StudentsPerformance.csv')
            logging.info('read the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)
            logging.info('Train test split initiated')

            train_set, test_set =train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
'''  
if __name__ == "__main__":
    obj=DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
    # model_trainer.initiate_model_trainer(train_arr, test_arr)
'''

if __name__ == "__main__":
    try:
        # Initialize and execute data ingestion
        logging.info(">>> Starting data ingestion <<<")
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        
        # Initialize and execute data transformation
        logging.info(">>> Starting data transformation <<<")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
        
        # Initialize and execute model training
        logging.info(">>> Starting model training <<<")
        model_trainer = ModelTrainer()
        results = model_trainer.initiate_model_trainer(train_arr, test_arr)
        
        logging.info(f">>> Training complete. Best model: {results['best_model']} <<<")
        print(f"\nTraining Results:\n{'-'*30}")
        print(f"Best Model: {results['best_model']}")
        print(f"Test R2 Score: {results['metrics']['test_metrics']['r2']:.4f}")
        print(f"Test MAE: {results['metrics']['test_metrics']['mae']:.4f}")
        
    except Exception as e:
        logging.error(f">>> Pipeline failed: {str(e)} <<<")
        raise CustomException(e, sys)
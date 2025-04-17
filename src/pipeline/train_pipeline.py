import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            logging.info("Starting training pipeline")
            
            # 1. Data Ingestion
            logging.info("Data Ingestion started")
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()
            
            # 2. Data Transformation
            logging.info("Data Transformation started")
            train_arr, test_arr, _ = self.data_transformation.initiate_data_transformation(train_path, test_path)
            
            # 3. Model Training
            logging.info("Model Training started")
            r2_score = self.model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info(f"Training completed. Best model R2 score: {r2_score}")
            
            return r2_score
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise CustomException(e, sys)
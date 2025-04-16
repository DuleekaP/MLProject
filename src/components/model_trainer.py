import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import logging
from time import time

# Model imports
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from scipy.stats import randint, uniform
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    explained_variance_score
)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    training_report_path: str = os.path.join('artifacts', 'training_report.csv')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def get_model_configs(self) -> Dict[str, Tuple[Any, Dict]]:
        """Define models and their hyperparameter search spaces with proper bounds"""
        return {
            # Existing models
            "RandomForest": (
                RandomForestRegressor(),
                {
                    'n_estimators': randint(50, 500),
                    'max_depth': [None, 5, 10, 20, 30],
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10)
                }
            ),
            "XGBoost": (
                XGBRegressor(),
                {
                    'n_estimators': randint(50, 500),
                    'learning_rate': uniform(0.01, 0.3),
                    'max_depth': randint(3, 10),
                    'colsample_bytree': uniform(0.6, 0.3)
                }
            ),
            "GradientBoosting": (
                GradientBoostingRegressor(),
                {
                    'n_estimators': randint(50, 500),
                    'learning_rate': uniform(0.01, 0.3),
                    'max_depth': randint(3, 10),
                    'subsample': uniform(0.6, 0.3)
                }
            ),
            "CatBoost": (
                CatBoostRegressor(verbose=0),
                {
                    'iterations': randint(50, 500),
                    'learning_rate': uniform(0.01, 0.3),
                    'depth': randint(3, 10)
                }
            ),
            
            # Updated models with fixes
            "LinearRegression": (
                LinearRegression(),
                {
                    'fit_intercept': [True, False],
                    'positive': [True, False]
                }
            ),
            "Ridge": (
                Ridge(),
                {
                    'alpha': uniform(0.1, 10),
                    'fit_intercept': [True, False],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
                }
            ),
            "Lasso": (
                Lasso(max_iter=5000),  # Increased max_iter to help convergence
                {
                    'alpha': uniform(0.1, 10),
                    'fit_intercept': [True, False],
                    'selection': ['cyclic', 'random']
                }
            ),
            "KNeighborsRegressor": (
                KNeighborsRegressor(),
                {
                    'n_neighbors': randint(3, 20),
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                }
            ),
            "DecisionTree": (
                DecisionTreeRegressor(),
                {
                    'max_depth': [None, 5, 10, 20, 30],
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10),
                    'max_features': [None, 'sqrt', 'log2'] 
                }
            )
        }
    def evaluate_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        models: Dict[str, Any],
        cv: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive model evaluation with cross-validation and test metrics
        
        Returns:
            Dictionary containing:
            - cv_metrics: Cross-validation results
            - test_metrics: Performance on holdout test set
            - training_time: Time taken for training
        """
        report = {}
        scoring = {
            'r2': 'r2',
            'neg_mae': 'neg_mean_absolute_error',
            'neg_rmse': 'neg_root_mean_squared_error'
        }

        for model_name, model in models.items():
            try:
                model_report = {}
                start_time = time()
                
                # Cross-validation
                cv_results = cross_validate(
                    model, X_train, y_train,
                    cv=cv, scoring=scoring,
                    return_train_score=True,
                    n_jobs=-1
                )
                
                # Test set evaluation
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Timing
                training_time = time() - start_time
                
                # Compile metrics
                model_report['cv_results'] = {
                    'mean_train_r2': np.mean(cv_results['train_r2']),
                    'mean_test_r2': np.mean(cv_results['test_r2']),
                    'mean_test_mae': -np.mean(cv_results['test_neg_mae']),
                    'mean_test_rmse': -np.mean(cv_results['test_neg_rmse'])
                }
                
                model_report['test_metrics'] = {
                    'r2': r2_score(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mape': mean_absolute_percentage_error(y_test, y_pred),
                    'explained_variance': explained_variance_score(y_test, y_pred)
                }
                
                model_report['training_time'] = training_time
                report[model_name] = model_report
                
                logging.info(f"{model_name} evaluation completed")
                
            except Exception as e:
                logging.error(f"Error evaluating {model_name}: {str(e)}")
                raise CustomException(e, sys)
                
        return report

    def tune_hyperparameters(self, model, params: Dict, X: np.ndarray, y: np.ndarray):
        try:
            # Calculate total parameter combinations
            total_params = 1
            for v in params.values():
                if hasattr(v, 'rvs'):  # It's a distribution
                    total_params *= 10  # Approximate for distributions
                else:
                    total_params *= len(v)
            
            # Adjust n_iter if parameter space is small
            n_iter = min(50, total_params) if total_params > 1 else 1
            
            search = RandomizedSearchCV(
                model, params,
                n_iter=n_iter,
                cv=3,
                scoring='r2',
                n_jobs=-1,
                random_state=42,
                error_score='raise'
            )
            search.fit(X, y)
            return search.best_estimator_, search.best_score_
        except Exception as e:
            logging.error(f"Hyperparameter tuning failed: {str(e)}")
            raise CustomException(e, sys)

    def initiate_model_trainer(
        self,
        train_arr: np.ndarray,
        test_arr: np.ndarray
    ) -> Dict[str, Any]:
        """
        Complete model training pipeline:
        1. Data splitting
        2. Hyperparameter tuning
        3. Model evaluation
        4. Best model selection
        5. Saving artifacts
        """
        try:
            logging.info("Splitting training and test data")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            
            # Get model configurations
            model_configs = self.get_model_configs()
            tuned_models = {}
            
            # Phase 1: Hyperparameter Tuning
            logging.info("Starting hyperparameter tuning")
            for model_name, (model, params) in model_configs.items():
                best_model, best_score = self.tune_hyperparameters(
                    model, params, X_train, y_train
                )
                tuned_models[model_name] = best_model
                logging.info(
                    f"Tuned {model_name} | Best R2: {best_score:.4f}"
                )
            
            # Phase 2: Comprehensive Evaluation
            logging.info("Starting model evaluation")
            evaluation_report = self.evaluate_models(
                X_train, y_train, X_test, y_test, tuned_models
            )
            
            # Phase 3: Best Model Selection
            best_model_name = max(
                evaluation_report.items(),
                key=lambda x: x[1]['test_metrics']['r2']
            )[0]
            best_model = tuned_models[best_model_name]
            best_metrics = evaluation_report[best_model_name]
            
            logging.info(f"""
            Best Model: {best_model_name}
            Test R2: {best_metrics['test_metrics']['r2']:.4f}
            Test MAE: {best_metrics['test_metrics']['mae']:.4f}
            Training Time: {best_metrics['training_time']:.2f}s
            """)
            
            # Phase 4: Save Artifacts
            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            # Save full report
            report_df = pd.DataFrame.from_dict({
                model: metrics['test_metrics']
                for model, metrics in evaluation_report.items()
            }, orient='index')
            report_df.to_csv(self.model_trainer_config.training_report_path)
            
            return {
                'best_model': best_model_name,
                'metrics': best_metrics,
                'full_report': evaluation_report
            }
            
        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")
            raise CustomException(e, sys)
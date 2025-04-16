# MLProject: Predictive Modeling Pipeline

## 📌 Overview
An end-to-end machine learning pipeline featuring:
- Data ingestion and preprocessing
- Advanced model training (Lasso, Gradient Boosting, XGBoost, etc.)
- Hyperparameter tuning with RandomizedSearchCV
- Model evaluation and selection

## 🏆 Best Performing Model
**Lasso Regression** achieved:
- R²: 0.882
- MAE: 4.165
- RMSE: 5.354

## 🛠️ Installation
```bash
git clone https://github.com/DuleekaP/MLProject.git
cd MLProject
pip install -r requirements.txt
```
##Usage
1. Run the pipeline:

python src/components/data_ingestion.py
python src/components/model_trainer.py


2. Expected output:

Training Results:
------------------------------
Best Model: Lasso
Test R2 Score: 0.882
Test MAE: 4.165

##Project Structure

.
├── artifacts/              # Saved models and reports
├── data/                   # Raw and processed data
├── notebooks/              # Exploratory analysis
├── src/
│   ├── components/         # Pipeline stages
│   ├── utils.py            # Helper functions
│   └── exception.py        # Custom exceptions
└── requirements.txt


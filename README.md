# MLProject: Predictive Modeling Pipeline

## 📌 Overview
A comprehensive machine learning pipeline for predicting student math scores based on demographic and academic factors, featuring:

- **Data Ingestion**: Automated train-test splitting and data storage
- **Data Transformation**: Advanced preprocessing (scaling, encoding, imputation)
- **Model Training**: Multiple algorithms with hyperparameter tuning
- **Prediction API**: Flask web interface for real-time predictions

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
python src/components/data_transformation.py 
python src/components/model_trainer.py

2. Launch the prediction web app:

python app.py

Visit http://localhost:5001 in your browser

to generate through API
http://localhost:5001/api/predict
example JSON request
    {
        "gender": "female",
        "race_ethnicity": "group A",
        "parental_level_of_education": "some college",
        "lunch": "standard",
        "test_preparation_course": "none",
        "reading_score": 99,
        "writing_score": 99
    }
example JSON response
    {
        "predicted_score": 90.6
    }

##Project Folder Structure

MLProject/
├── artifacts/               # Serialized models and preprocessors
│   ├── model.pkl            # Trained model
│   └── preprocessor.pkl     # Data transformation pipeline
├── data/                    # Raw and processed datasets
├── notebooks/               # Jupyter notebooks for EDA
├── src/
│   ├── components/          # Pipeline stages
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/            # Prediction workflow
│   │   └── predict_pipeline.py
│   ├── exception.py         # Custom exception handling
│   └── utils.py             # Helper functions
├── templates/               # Flask HTML templates
│   ├── home.html            # Prediction form
│   └── index.html           # Landing page
├── app.py                   # Flask application
└── requirements.txt         # Dependencies

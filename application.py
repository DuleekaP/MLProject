from flask import Flask, request, render_template, redirect, url_for
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])  # Add GET method here
def predict():
    if request.method == 'POST':
        try:
            # Validate required fields
            reading_score = request.form.get('reading_score')
            writing_score = request.form.get('writing_score')
            
            if not reading_score or not writing_score:
                return "Reading and Writing scores are required", 400

            try:
                reading_score = int(reading_score)
                writing_score = int(writing_score)
            except ValueError:
                return "Scores must be numeric values", 400

            if not (0 <= reading_score <= 100) or not (0 <= writing_score <= 100):
                return "Scores must be between 0 and 100", 400

            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=reading_score,
                writing_score=writing_score
            )

            pred_df = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            if round(results[0]) > 100:
                results[0] = 100
            return render_template('home.html', results=round(results[0],2))
        
        except Exception as e:
            return f"An error occurred: {str(e)}", 500
    else:
        # Handle GET request - just show the form
        return render_template('home.html')
    
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        reading_score = data.get('reading_score')
        writing_score = data.get('writing_score')

        if reading_score is None or writing_score is None:
            return {"error": "Reading and Writing scores are required"}, 400

        try:
            reading_score = int(reading_score)
            writing_score = int(writing_score)
        except ValueError:
            return {"error": "Scores must be numeric values"}, 400

        if not (0 <= reading_score <= 100) or not (0 <= writing_score <= 100):
            return {"error": "Scores must be between 0 and 100"}, 400

        custom_data = CustomData(
            gender=data.get('gender'),
            race_ethnicity=data.get('race_ethnicity'),
            parental_level_of_education=data.get('parental_level_of_education'),
            lunch=data.get('lunch'),
            test_preparation_course=data.get('test_preparation_course'),
            reading_score=reading_score,
            writing_score=writing_score
        )

        pred_df = custom_data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        predicted_score = round(results[0], 2)
        if predicted_score > 100:
            predicted_score = 100
        
        return {"predicted_score": predicted_score}, 200

    except Exception as e:
        return {"error": str(e)}, 500
if __name__ == "__main__":
    app.run(host="0.0.0.0")
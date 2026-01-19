import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
import io

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load Model and Pipeline
MODEL_FILE = "Model/model.pkl"
PIPELINE_FILE = "Model/pipeline.pkl"

model = None
pipeline = None

def load_artifacts():
    global model, pipeline
    if os.path.exists(MODEL_FILE) and os.path.exists(PIPELINE_FILE):
        model = joblib.load(MODEL_FILE)
        pipeline = joblib.load(PIPELINE_FILE)
        print("Model and Pipeline loaded successfully.")
    else:
        print("Error: Model or Pipeline file not found. Please train the model first.")

load_artifacts()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not pipeline:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        
        # Convert single item to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess using loaded pipeline
        # The pipeline expects specific columns. We rely on the frontend to send correct keys.
        # We need to handle 'Price' and 'Price_log' leakage prevention if the pipeline expects them (it shouldn't based on training script logic)
        # However, the training script drops Price/Price_log before pipeline fitting.
        
        # Ensure numeric conversion for numeric fields
        numeric_cols = ['Levy', 'Engine volume', 'Mileage(Km)', 'Cylinders', 'Airbags', 'Age']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Transform data
        # Note: The pipeline relies on named steps 'num' and 'cat' handling specific columns.
        # We trust the pipeline to select the columns it needs from the dataframe.
        
        # Manually imputation for numeric columns if needed (pipeline handles this usually)
        # But pipeline.transform() expects the exact same structure?
        # Scikit-learn ClusterTransformer selects columns if configured with column names.
        # Our training script build_pipeline uses lists of column names.
        
        data_prepared = pipeline.transform(df)
        prediction_log = model.predict(data_prepared)
        
        # Inverse log transform if model predicted log price
        # Training script: 
        # train_set["Price_log"]=np.log1p(train_set["Price"])
        # model.fit(data_prepared, data_labels) -> data_labels was train_set["Price"]
        # WAIT. Line 50 in training script: data_labels=train_set["Price"].copy()
        # It trains on RAW PRICE, not Log Price. 
        # Line 49 creates Price_log but Line 50 uses Price.
        # Line 63 model.fit(..., data_labels)
        # So output is direct Price.
        
        predicted_price = prediction_log[0]
        
        return jsonify({'prediction': f"{predicted_price:,.2f}"})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if not model or not pipeline:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        df = pd.read_csv(file)
        
        # Basic Validation: Ensure some columns exist
        required_cols = ["Manufacturer", "Model"] # Check at least these
        pass_check = True
        for col in required_cols:
            if col not in df.columns:
                pass_check = False
        
        # Preprocessing
        # Note: We must ensure not to pass Price column to pipeline transformation if it exists in upload
        df_for_pred = df.copy()
        if "Price" in df_for_pred.columns:
            df_for_pred = df_for_pred.drop("Price", axis=1)
            
        data_prepared = pipeline.transform(df_for_pred)
        predictions = model.predict(data_prepared)
        
        df['Predicted_Price'] = predictions
        
        # Save to buffer
        output = io.BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(output, mimetype='text/csv', as_attachment=True, download_name='predictions.csv')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

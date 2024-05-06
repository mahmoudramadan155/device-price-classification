from flask import Flask, request, jsonify
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model_svm = joblib.load('svm_device_price_classifier.pkl')

# Preprocessing steps
imputer = SimpleImputer(strategy='constant', fill_value=0)
scaler = StandardScaler()

# Define endpoint for price prediction
@app.route('/predict', methods=['POST'])
def predict_price():
    data = request.json
    device_specs = data.get('specs')  # Assuming the request contains device specifications
    
    if device_specs is None:
        return jsonify({'error': 'No device specifications provided'}), 400
    
    try:
        # Perform preprocessing
        device_specs = imputer.transform([device_specs])
        device_specs = scaler.transform(device_specs)
        
        # Predict the price using the loaded model
        predicted_price = model_svm.predict(device_specs)[0]
        
        # Map predicted price to price range
        price_range_mapping = {0: 'low cost', 1: 'medium cost', 2: 'high cost', 3: 'very high cost'}
        predicted_price_range = price_range_mapping[predicted_price]
        
        return jsonify({'predicted_price_range': predicted_price_range})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

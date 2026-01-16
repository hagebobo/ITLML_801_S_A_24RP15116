"""
Heart Disease Risk Prediction System - Flask REST API
Student ID: 24RP15116

This Flask application provides a REST API for heart disease risk prediction.
Based on the Jupyter notebook implementation.
"""

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Initialize Flask app
app = Flask(__name__)

# Model and artifact paths
MODEL_PATH = 'deployment/heart_disease_model_24RP15116.pkl'
FEATURE_NAMES_PATH = 'deployment/feature_columns.txt'
CLASS_NAMES_PATH = 'deployment/class_names.txt'

# Global variables
model = None
feature_columns = None
class_names = None

def load_model_artifacts():
    """Load all model artifacts at startup"""
    global model, feature_columns, class_names
    
    print("\n" + "="*80)
    print(" LOADING MODEL ARTIFACTS...")
    print("="*80)
    
    try:
        # Check files exist
        print(f"\n Checking file existence...")
        print(f"  Model: {MODEL_PATH} -> {'✓ EXISTS' if os.path.exists(MODEL_PATH) else '✗ NOT FOUND'}")
        print(f"  Features: {FEATURE_NAMES_PATH} -> {'✓ EXISTS' if os.path.exists(FEATURE_NAMES_PATH) else '✗ NOT FOUND'}")
        print(f"  Classes: {CLASS_NAMES_PATH} -> {'✓ EXISTS' if os.path.exists(CLASS_NAMES_PATH) else '✗ NOT FOUND'}")
        
        # Load the full pipeline (preprocessor + classifier)
        print(f"\n Loading pipeline from {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
        print(f"✓ Pipeline loaded successfully!")
        print(f"  Type: {type(model)}")
        
        # Load feature column names (plain text, one per line)
        print(f"\n Loading feature columns from {FEATURE_NAMES_PATH}...")
        with open(FEATURE_NAMES_PATH, 'r') as f:
            feature_columns = [line.strip() for line in f if line.strip()]
        print(f"✓ Feature columns loaded!")
        print(f"  Features: {feature_columns}")
        print(f"  Total: {len(feature_columns)}")
        
        # Load class names (plain text, one per line)
        print(f"\n  Loading class names from {CLASS_NAMES_PATH}...")
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
        print(f"✓ Class names loaded!")
        print(f"  Classes: {class_names}")
        
        print("\n" + "="*80)
        print(" ALL ARTIFACTS LOADED SUCCESSFULLY!")
        print("="*80 + "\n")
        return True
        
    except Exception as e:
        print("\n" + "="*80)
        print(" ERROR LOADING ARTIFACTS!")
        print("="*80)
        print(f"Error: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())
        print("="*80 + "\n")
        return False

# Load artifacts at startup
load_success = load_model_artifacts()

# Risk level mapping (based on notebook classes)
RISK_LEVELS = {
    0: {"color": "#2ecc71", "label": "No Disease", "description": "Patient shows no signs of heart disease"},
    1: {"color": "#3498db", "label": "Very Mild", "description": "Very mild heart condition detected, monitor regularly"},
    2: {"color": "#f39c12", "label": "Mild", "description": "Mild heart condition, medical consultation recommended"},
    3: {"color": "#e74c3c", "label": "Severe", "description": "Severe heart condition, immediate medical attention needed"},
    4: {"color": "#8e44ad", "label": "Immediate Danger", "description": "Critical condition, emergency medical intervention required"}
}

@app.route('/')
def home():
    """Render the main HTML interface"""
    print("\n Home page requested")
    return render_template('index_24RP15116.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    print("\n Health check requested")
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'features_loaded': feature_columns is not None,
        'classes_loaded': class_names is not None,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    print(f"  Status: {status}")
    return jsonify(status), 200

@app.route('/api/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    print("\n" + "="*80)
    print("  API INFO REQUEST")
    print("="*80)
    
    try:
        if model is None or feature_columns is None or class_names is None:
            print(" Some artifacts are None!")
            return jsonify({
                'status': 'error',
                'message': 'Model artifacts not fully loaded'
            }), 500
        
        response = {
            'status': 'success',
            'model_info': {
                'name': 'Heart Disease Classifier Pipeline',
                'type': 'Sklearn Pipeline (Preprocessor + Classifier)',
                'version': '1.0.0',
                'student_id': '24RP15116'
            },
            'features': {
                'names': feature_columns,
                'count': len(feature_columns)
            },
            'classes': {
                'names': class_names,
                'count': len(class_names)
            },
            'risk_levels': RISK_LEVELS
        }
        print("Response prepared successfully")
        return jsonify(response), 200
        
    except Exception as e:
        print(f" Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving model information: {str(e)}'
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make heart disease risk prediction"""
    print("\n" + "="*80)
    print(" PREDICTION REQUEST RECEIVED")
    print("="*80)
    
    try:
        # Check artifacts loaded
        print(f"\n Checking artifacts:")
        print(f"  Model: {model is not None}")
        print(f"  Features: {feature_columns is not None}")
        print(f"  Classes: {class_names is not None}")
        
        if model is None:
            print(" Model is None!")
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 500
        
        if feature_columns is None:
            print(" Feature columns are None!")
            return jsonify({
                'status': 'error',
                'message': 'Feature columns not loaded'
            }), 500
        
        if class_names is None:
            print(" Class names are None!")
            return jsonify({
                'status': 'error',
                'message': 'Class names not loaded'
            }), 500
        
        # Get request data
        print("\n Getting request data...")
        data = request.get_json()
        
        if not data:
            print(" No data received!")
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        print(f" Data received:")
        for key, value in data.items():
            print(f"  {key}: {value}")
        
        # Validate features
        print(f"\n Validating features...")
        print(f"  Expected features: {feature_columns}")
        missing = [f for f in feature_columns if f not in data]
        
        if missing:
            print(f" Missing features: {missing}")
            return jsonify({
                'status': 'error',
                'message': f'Missing required features: {", ".join(missing)}'
            }), 400
        
        print("All features present")
        
        # Create DataFrame in correct order
        print("\n Creating input DataFrame...")
        # Convert boolean string to actual boolean for 'fbs' if needed
        if 'fbs' in data and isinstance(data['fbs'], str):
            data['fbs'] = data['fbs'].lower() == 'true'
        
        input_df = pd.DataFrame([data], columns=feature_columns)
        print(f"  Shape: {input_df.shape}")
        print(f"  Columns: {list(input_df.columns)}")
        print(f"  Data types:")
        for col in input_df.columns:
            print(f"    {col}: {input_df[col].dtype} = {input_df[col].values[0]}")
        
        # Make prediction using the pipeline
        # The pipeline handles all preprocessing internally
        print("\n Making prediction...")
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        print(f"✓ Prediction made:")
        print(f"  Predicted class: {prediction}")
        print(f"  Class label: {class_names[prediction]}")
        print(f"  Probabilities: {probabilities}")
        
        # Get risk information
        risk_info = RISK_LEVELS[int(prediction)]
        
        # Create probability distribution
        prob_distribution = []
        for i, prob in enumerate(probabilities):
            prob_distribution.append({
                'class': i,
                'label': class_names[i],
                'probability': float(prob),
                'percentage': float(prob * 100),
                'color': RISK_LEVELS[i]['color']
            })
        
        # Sort by probability
        prob_distribution.sort(key=lambda x: x['probability'], reverse=True)
        
        # Determine confidence
        max_prob = float(max(probabilities))
        if max_prob >= 0.8:
            confidence = "High"
        elif max_prob >= 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        print(f"  Confidence: {confidence} ({max_prob:.2%})")
        
        # Generate recommendations
        recommendations = generate_recommendations(int(prediction), max_prob)
        
        # Create response
        response = {
            'status': 'success',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'prediction': {
                'class': int(prediction),
                'label': risk_info['label'],
                'description': risk_info['description'],
                'color': risk_info['color'],
                'confidence': confidence,
                'confidence_score': max_prob
            },
            'probabilities': prob_distribution,
            'input_data': data,
            'recommendations': recommendations
        }
        
        print("\n PREDICTION SUCCESSFUL!")
        print("="*80 + "\n")
        return jsonify(response), 200
        
    except Exception as e:
        print("\n PREDICTION ERROR!")
        print("="*80)
        print(f"Error: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())
        print("="*80 + "\n")
        return jsonify({
            'status': 'error',
            'message': f'Prediction error: {str(e)}'
        }), 500

def generate_recommendations(prediction_class, confidence):
    """Generate recommendations based on prediction"""
    recommendations = []
    
    if prediction_class == 0:  # No Disease
        recommendations = [
            " Maintain healthy lifestyle with regular exercise",
            " Continue balanced diet and monitor vitals regularly",
            " Annual health checkup recommended"
        ]
    elif prediction_class == 1:  # Very Mild
        recommendations = [
            " Schedule consultation with cardiologist",
            " Monitor blood pressure and cholesterol levels",
            " Consider lifestyle modifications",
            " Regular follow-up every 6 months"
        ]
    elif prediction_class == 2:  # Mild
        recommendations = [
            " Medical consultation strongly recommended",
            " Complete cardiac evaluation needed",
            " Monitor symptoms closely",
            " Follow prescribed medication regimen",
            " Regular follow-up every 3 months"
        ]
    elif prediction_class == 3:  # Severe
        recommendations = [
            " URGENT: Seek immediate medical attention",
            " Comprehensive cardiac workup required",
            " May require hospitalization",
            " Strict medication compliance essential",
            " Monthly follow-up mandatory"
        ]
    else:  # Immediate Danger
        recommendations = [
            " CRITICAL: EMERGENCY MEDICAL INTERVENTION REQUIRED",
            " Call emergency services immediately",
            " Do not delay - life-threatening condition",
            " Hospital admission necessary",
            " Intensive cardiac care needed"
        ]
    
    if confidence < 0.7:
        recommendations.append("ℹ Note: Moderate confidence. Clinical judgment essential.")
    
    return recommendations

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    print(f"\n 404 Error: {request.url}")
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    print(f"\n 500 Error: {str(error)}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("  HEART DISEASE RISK PREDICTION SYSTEM - REST API")
    print("   Student ID: 24RP15116")
    print("="*80)
    
    if not load_success:
        print("\n  WARNING: Some artifacts failed to load!")
        print("   The application will start but predictions may not work.")
    
    print("\n Available endpoints:")
    print("  - GET  /              : Main web interface")
    print("  - GET  /api/info      : Model information")
    print("  - POST /api/predict   : Make prediction")
    print("  - GET  /api/health    : Health check")
    print("\n Server starting on: http://127.0.0.1:5000")
    print("="*80 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
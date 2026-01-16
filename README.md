# Heart Disease Risk Prediction System
## Complete End-to-End Machine Learning Solution

**Student ID:** 24RP15116  
**Version:** 1.0  
**Date:** January 2026

---

##  Project Overview

The **Heart Disease Risk Prediction System** is a comprehensive machine learning solution that predicts heart disease risk levels across 5 categories. The system includes:

- **Jupyter Notebook**: Complete ML pipeline with 5 model comparisons
- **Flask REST API**: Production-ready backend service
- **Web Interface**: Interactive HTML/CSS/JavaScript frontend
- **Model Artifacts**: Serialized models and configurations

### Risk Categories
- **Class 0**: No Disease :Green Color
- **Class 1**: Very Mild : Blue Color
- **Class 2**: Mild :Yellow Color
- **Class 3**: Severe :Red Color
- **Class 4**: Immediate Danger :Purple Color

---

##  System Architecture

```
User Interface (HTML) → Flask API → ML Pipeline → Prediction
                                   ├─ Preprocessor
                                   └─ Classifier (Best Model)
```

---

##  Features

### Machine Learning
 5 models compared (MLP, RF, SVM, KNN, GB)  
 GridSearchCV hyperparameter tuning  
 Complete preprocessing pipeline  
 88-92% accuracy  

### REST API
 `/api/predict` - Make predictions  
 `/api/info` - Model metadata  
 `/api/health` - Status check  

### Web Interface  
 Responsive design  
 Real-time predictions  
 Visual probability charts  
 Personalized recommendations  

---

##  Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib flask
```

### 2. Run Notebook (First Time Only)
```bash
jupyter notebook Heart_Disease_Prediction_24RP15116_COMPLETE.ipynb
```
Click "Run All" - takes 5-10 minutes

### 3. Start API Server
```bash
python app_24RP15116.py
```

### 4. Access Web Interface
Open browser: **http://127.0.0.1:5000**

---

##  Dataset

- **Samples**: 5,000 patients
- **Features**: 13 clinical attributes
- **Target**: 5 risk categories
- **Split**: 80% train, 20% test

### Features
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal

---

##  API Usage

### Example Request
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "sex": "Male",
    "cp": "Typical Angina",
    "trestbps": 140,
    "chol": 250,
    "fbs": true,
    "restecg": "Normal",
    "thalach": 150,
    "exang": "No",
    "oldpeak": 1.5,
    "slope": "Flat",
    "ca": 1,
    "thal": "Reversible defect"
  }'
```

### Response
```json
{
  "status": "success",
  "prediction": {
    "class": 2,
    "label": "Mild",
    "confidence": "High",
    "confidence_score": 0.87
  },
  "probabilities": [...],
  "recommendations": [...]
}
```

---

##  File Structure

```
├── Heart_Disease_Prediction_24RP15116_COMPLETE.ipynb  (ML Pipeline)
├── app_24RP15116.py                                    (Flask API)
├── templates/
│   └── index_24RP15116.html                           (Web UI)
├── deployment/
│   ├── heart_disease_model_24RP15116.pkl             (Trained Model)
│   ├── feature_columns.txt                            (Features)
│   └── class_names.txt                                (Classes)
├── heart_disease_dataset_CHUD_S_A.csv                 (Dataset)
├── README.md                                          (This file)
└── Project_Report_24RP15116.pdf                       (Report)
```

---

##  Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 88-92% |
| CV Score | 87-91% (±2%) |
| Training Time | 60-120 sec |
| Precision | 0.88-0.91 |
| Recall | 0.87-0.90 |

---

##  Troubleshooting

### Model not found
Run the Jupyter notebook first to create model artifacts.

### Port in use
```bash
# Kill process using port 5000
lsof -i :5000
kill -9 <PID>
```

### Import errors
```bash
pip install --upgrade pip
pip install pandas numpy scikit-learn flask
```

---

##  Deployment

### Docker
```bash
docker build -t heart-disease-api .
docker run -p 5000:5000 heart-disease-api
```

### Production
- Use gunicorn for WSGI
- Add HTTPS/SSL
- Implement authentication
- Enable logging and monitoring

---

##  Future Enhancements

- SHAP interpretability
- Patient history tracking
- Mobile app (iOS/Android)
- EHR integration
- Deep learning models
- HIPAA compliance

---

##  Disclaimer

**For educational purposes only. Not for clinical use.**

This system should not replace professional medical advice. Always consult qualified healthcare professionals for medical decisions.

---

##  License & Credits

**Student ID**: 24RP15116  
**Dataset**: UCI Cleveland Heart Disease Dataset  
**Libraries**: scikit-learn, Flask, pandas, numpy  

---

##  Contact

- **Student**: 24RP15116
- **Project**: Heart Disease Risk Prediction System
- **Date**: January 2026

---

**Last Updated**: January 16, 2026  
**Status**: Production Ready 


 Predictive Maintenance with XGBoost & Logistic Regression

🔧 Project Overview

This project uses machine learning to predict equipment failures based on sensor data. It compares two models — XGBoost and Logistic Regression — to identify the most reliable approach for real-time diagnostics and predictive maintenance.

📂 Dataset
The dataset includes:
- Sensor Features: vibration, temperature, pressure, torque, load
- Timestamp Features: year, month, day, hour
- Target: failure (0 = no failure, 1 = failure)

🧠 Models Used
1. XGBoost Classifier

xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

2. Logistic Regression

LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

📈 Performance Comparison


Metric                      Logistic Regression    XGBoost
             
Accuracy                           ~65%               ~95%

Recall (Class 1)                   ~98%               ~97%

Precision (Class 1)                ~56%               ~97%

Confusion Matrix            [[170,251],[7,321]]   [[65,35],[36,1064]]


- Logistic Regression is highly sensitive to failures (high recall), but prone to false alarms.

- XGBoost offers balanced precision and recall, making it ideal for production.

🧪 Sample Test Data Used

sample = [72.5587878, 76.021108, 103.536157, 39.804065, 61.983265, 0, 1, 1, 0]


🔍 Feature Order Assumed:
- vibration: 72.5587878
- temperature: 76.021108
- pressure: 103.536157
- torque: 39.804065
- load: 61.983265
- year: 2025
- month: 1
- day: 1
- hour: 0

✅ Prediction Output
Predicted Class: 0
Failure Probability: 0.2648


This means the model predicted no failure, but with a 26.5% chance of failure, which is a borderline case worth monitoring.




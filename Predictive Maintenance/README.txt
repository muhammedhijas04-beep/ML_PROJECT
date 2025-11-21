🔧 Predictive Maintenance Using XGBoost & Logistic Regression

A Machine Learning Approach for Failure Prediction in Industrial Equipment.

📌 Overview

This project builds a predictive maintenance model using real sensor data to identify early signs of machine failure.

Two ML models are compared:

XGBoost Classifier → advanced, tree-based, excellent on imbalanced sensor data

Logistic Regression → linear baseline with strong recall

The goal is to determine which model is more reliable for real-time fault detection.

📂 Dataset Description

The dataset contains continuous sensor readings collected from industrial equipment:

Sensor Features

vibration

temperature

pressure

torque

load

Time Features

Extracted from timestamp:

year, month, day, hour

Target Variable

failure → 0 = normal, 1 = failure

🧠 Models Implemented

1️⃣ XGBoost Classifier

xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=1,
    eval_metric='logloss',
    random_state=42
)

✔ Handles nonlinear relationships
✔ More stable on imbalanced data
✔ High precision + high recall

2️⃣ Logistic Regression

LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

✔ Excellent baseline model
✔ Very sensitive to failures (high recall)
✔ Simpler and easier to interpret

📈 Model Performance Comparison

| Metric                        | Logistic Regression      | XGBoost                  |
| ----------------------------- | ------------------------ | ------------------------ |
|   Accuracy                    |  65%                     |  95%                     |
|   Recall (Failure Class)      |  98%                     |  97%                     |
|   Precision (Failure Class)   |  56%                     |  97%                     |
|   Confusion Matrix            |  [[170, 251], [7, 321]]  |  [[65, 35], [36, 1064]]  |

🔍 Interpretation

Logistic Regression

Best when catching every failure is critical

High recall but triggers many false alarms

XGBoost

Strong balance of precision + recall

Ideal for production environments

Fewer false positives and high confidence predictions

🚀 How to Run the Project

pip install xgboost pandas scikit-learn matplotlib

X = df.drop("failure", axis=1)
y = df["failure"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression

model = LogisticRegression(max_iter=1000,class_weight ="balanced")
model.fit(x_train,y_train)
prediction =model.predict(x_test)

# XGBoost

model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1,max_depth=5, scale_pos_weight=1, use_label_encoder=False, eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train) 

Predict on a sample

(sample = [[vibration, temperature, pressure, torque, load, year, month, day, hour]])
sample = [[72.557878, 76.021108, 103.536157, 39.804065, 61.983265, 0, 1, 1, 0]]

prediction = model.predict(sample)
proba = model.predict_proba(sample)

print("Predicted Class:", prediction[0])
print("Failure Probability:", proba[0][1])

Predicted Class: 0
Failure Probability: 0.26477787


📊 Optional Enhancements (Future Work)

Threshold tuning using probability outputs

XGBoost feature importance visualization

ROC-AUC comparison between models

Real-time inference pipeline or dashboard

Deployment using FastAPI or Streamlit

Muhammed Hijas
B.Tech Mechatronics | Data Science
Focusing on:

🛠️ Author

Predictive Maintenance

Industrial Sensor Analytics

Intelligent Automation

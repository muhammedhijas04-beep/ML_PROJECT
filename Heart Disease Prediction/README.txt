❤️ Heart Disease Prediction using Machine Learning

A machine learning project that predicts the presence of heart disease in patients using clinical parameters.  
This project compares two classification algorithms — **Logistic Regression** and **Random Forest Classifier** — to achieve accurate and interpretable predictions.



 🎯 Overview
Heart disease remains one of the most critical health issues globally.  
This project leverages **machine learning techniques** to predict whether a patient has heart disease based on clinical parameters such as blood pressure, cholesterol levels, and age.  
It assists in **early risk detection**, supporting healthcare professionals in better diagnosis and prevention.


 📊 Dataset
- Original Records: 1,025  
- Duplicate Rows: 723  
- Final Clean Dataset: 302 unique samples  
- Attributes:** 14  
- Target Variable: `target` → (1 = Heart Disease, 0 = No Heart Disease)

 Features
| Feature | Description |
|----------|-------------|
| age | Age of the patient |
| sex | Gender (1 = male, 0 = female) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| restecg | Resting electrocardiographic results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1 = yes, 0 = no) |
| oldpeak | ST depression induced by exercise relative to rest |
| slope | Slope of the peak exercise ST segment (0–2) |
| ca | Number of major vessels colored by fluoroscopy (0–4) |
| thal | Thalassemia (0–3) |
| target | Heart disease presence (1 = disease, 0 = no disease) |


 ✨ Key Features
- Comprehensive **exploratory data analysis (EDA)**  
- Removal of duplicate and inconsistent records  
- Feature correlation** and visualization  
- Implementation of **multiple ML models**  
- Performance comparison** across algorithms  
- Probability predictions** for risk scoring  



 🤖 Models Implemented

 1️⃣ Logistic Regression
- Max iterations: 1000
- Class weight: balanced

 2️⃣ Random Forest Classifier
- Number of estimators: 100

📈 Results

| Model               | Accuracy | Precision (Avg) | Recall (Avg) | F1-Score (Avg) |
| ------------------- | -------- | --------------- | ------------ | -------------- |
| Logistic Regression | 86%      | 0.86            | 0.86         | 0.86           |
| Random Forest       | 82%      | 0.84            | 0.83         | 0.82           |


💾 Output Example

prediction = model.predict([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
print(prediction)   # [0] → No Heart Disease



🔍 Insights

Logistic Regression performed best overall (86% accuracy)

Random Forest had higher recall for positive (disease) cases (93%)

Dataset is slightly balanced (~54% disease cases)

Models are reliable for basic clinical prediction tasks





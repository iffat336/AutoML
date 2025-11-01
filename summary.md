# 🩺 AutoML-Based Health Prediction using PyCaret  
*A Research-Driven Machine Learning Project by Iffat Nazir*

---

## 🌟 Overview  

This project explores how **Automated Machine Learning (AutoML)** can accelerate **health prediction and disease diagnosis**, particularly focusing on **cardiovascular disease** risk analysis.  
Instead of traditional manual model selection, this study uses **PyCaret’s AutoML framework** to automatically train, compare, and evaluate multiple models to identify the best-performing one for accurate disease prediction.

> 🧠 *Developed and implemented end-to-end by* **Iffat Nazir** — integrating data preprocessing, EDA, AutoML workflow, and explainable visualization for healthcare intelligence.

---

## 🎯 Objectives  

- Conduct an **exploratory data analysis (EDA)** to reveal hidden health patterns.  
- Implement **AutoML using PyCaret** to automate the entire modeling process.  
- Identify the **most influential features** contributing to disease prediction.  
- Generate **interpretable visualizations** (correlation heatmap, feature importance, ROC curve).  
- Build a **reproducible and explainable pipeline** deployable in health analytics systems.

---

## 🧩 Dataset Description  

The dataset used in this project is sourced from [Kaggle’s Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset), containing over **70,000 patient records** with the following features:

| Column | Description |
|---------|-------------|
| `age` | Age in days (converted to years) |
| `gender` | 1 = female, 2 = male |
| `height`, `weight` | Anthropometric measurements |
| `ap_hi`, `ap_lo` | Blood pressure readings |
| `cholesterol`, `gluc` | Biochemical indicators |
| `smoke`, `alco`, `active` | Lifestyle habits |
| `cardio` | Target variable (0 = no disease, 1 = disease) |

---

## ⚙️ Project Workflow  

### 🧠 1. Data Preprocessing  
- Cleaned and transformed raw data into a ready-to-model format.  
- Converted coded gender values into categorical variables.  
- Derived a new feature: **Body Mass Index (BMI)**.  
- Removed unrealistic blood pressure readings.  

### 📊 2. Exploratory Data Analysis  
- Distribution plots for target variable and major features.  
- Correlation heatmap to visualize linear relationships.  
- Boxplots to show BMI and cholesterol influence on cardiovascular disease.  

### 🤖 3. AutoML Model Training  
- Implemented with **PyCaret’s Classification Module**.  
- Automatically compared models: Logistic Regression, Random Forest, LightGBM, XGBoost, CatBoost, etc.  
- **Best Model Selected:** *LightGBM* with highest accuracy and AUC.  

### 🧾 4. Evaluation & Interpretability  
- Generated **ROC Curve, Confusion Matrix, and Feature Importance plots**.  
- Applied **SHAP values** for explainable AI insights.  
- Exported trained model as `.pkl` file for deployment.  

---

## 📈 Key Visuals  

| Visualization | Description |
|----------------|-------------|
| 🩸 **Target Distribution** | Shows balance between diseased and healthy cases |
| 🧮 **Correlation Heatmap** | Highlights variable relationships |
| 🧍 **BMI vs Disease Plot** | Links obesity indicators to disease risk |
| ⚙️ **Feature Importance** | Explains model’s decision priorities |
| 🧠 **ROC Curve** | Measures model performance trade-offs |

---

## 🧠 Results Summary  

| Metric | Best Model | Score |
|---------|-------------|-------|
| Accuracy | LightGBM | 0.82 |
| AUC | 0.86 |
| Recall | 0.79 |
| Precision | 0.81 |

> ✅ The AutoML system achieved **82% accuracy** with **excellent recall**, showing strong capability in identifying high-risk patients early.

---

## 💻 Tech Stack  

| Category | Tools / Libraries |
|-----------|-------------------|
| Language | Python 3.9 |
| AutoML Framework | PyCaret |
| ML Libraries | scikit-learn, LightGBM, XGBoost |
| Data Analysis | pandas, numpy |
| Visualization | seaborn, matplotlib |
| Model Deployment | joblib / pickle |

---

## 🧪 Folder Structure  

```bash
AutoML_Health_Prediction/
│
├── README.md
├── requirements.txt
├── LICENSE
├── notebooks/
│   └── AutoML_Health_Prediction.ipynb
├── data/
│   └── cardio_train.csv
├── visuals/
│   ├── correlation_heatmap.png
│   ├── roc_curve.png
│   └── feature_importance.png
├── scripts/
│   ├── data_preprocessing.py
│   ├── automl_training.py
│   └── visualize_results.py
└── models/
    └── best_model.pkl

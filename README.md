# 🧠 ChurnVision AI

> **An advanced AI-powered pipeline for customer retention intelligence.**  
> ChurnVision AI detects early churn patterns, uncovers behavioral clusters, and predicts customer attrition with explainable machine learning and ensemble models.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](#)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML--Modeling-orange)](#)
[![Pandas](https://img.shields.io/badge/Pandas-Data--Processing-green)](#)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-red)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#)

---

## 📚 Table of Contents
1. [Overview](#overview)
2. [Project Objectives](#project-objectives)
3. [Architecture](#architecture)
4. [Data & Preprocessing](#data--preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Modeling Approach](#modeling-approach)
7. [Evaluation & Results](#evaluation--results)
8. [Key Visual Insights](#key-visual-insights)
9. [How to Run](#how-to-run)
10. [Future Enhancements](#future-enhancements)
11. [License](#license)
12. [Author](#author)

---

## 🔍 Overview
Customer churn is one of the most significant challenges for any subscription-based business.  
**ChurnVision AI** provides a holistic solution that:
- Cleans and preprocesses raw customer data.  
- Identifies customer segments through clustering.  
- Mines association rules for behavioral dependencies.  
- Trains multiple ML models to predict churn probability.  
- Fuses them through a voting-based ensemble for accuracy and stability.  

The result is an **end-to-end explainable pipeline**—from data to actionable retention strategies.

---

## 🎯 Project Objectives
- **Predict churn** with high accuracy using multi-model learning.  
- **Visualize behavioral clusters** to understand why customers leave.  
- **Identify hidden correlations** between services and churn likelihood.  
- **Enable business decisions** through interpretable and reproducible analytics.

---

## 🧩 Architecture

<p align="center">
  <img src="./reports/pipeline_architecture.png" alt="Pipeline Overview" width="850">
</p>

**Figure 1 — Overall architecture:** data ingestion → preprocessing → modeling → ensemble → visualization.

---

## 🧹 Data & Preprocessing
- **Dataset:** Telco customer dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)  
- **Processed Output:** `new_telco.csv`  
- **Target Variable:** `Churn` (Yes / No)  

**Cleaning Steps**
- Converted categorical fields to numerical using one-hot encoding.  
- Coerced `TotalCharges` to numeric and handled missing values.  
- Standardized continuous variables (`tenure`, `MonthlyCharges`, `TotalCharges`).  
- Split data into 70 % training / 30 % testing (stratified).

---

## ⚙️ Feature Engineering
New derived features improve model interpretability:
- `AvgMonthlySpend = TotalCharges / tenure`  
- `TenureGroup` buckets: *0–12, 13–24, 25–48, 49+*  
- `HasMultipleServices` flag: customers with multiple active services  

Feature selection was guided by correlation matrices, permutation importance, and association rules.

---

## 🤖 Modeling Approach

The pipeline explores **five key learning modules**, each contributing unique insight:

| Module | Purpose | Output |
|:-------|:---------|:--------|
| **Decision Tree** | Establishes interpretable baseline | Feature importance map |
| **Clustering (K-Prototypes)** | Groups customers by behavioral similarity | Customer segments |
| **Association Rule Mining** | Finds frequently co-occurring attributes | Churn-related patterns |
| **Classifiers (LR, DT, KNN, NB, RF, SVC)** | Predict churn probability | Individual model metrics |
| **Voting Ensemble** | Combines all predictions | Final churn probability |

<p align="center">
  <img src="./reports/ID3graph.JPG" alt="Decision Tree Visualization" width="750">
</p>

**Figure 2 — Decision Tree:** contract type and tenure dominate churn decisions.

---

## 📊 Evaluation & Results

| Model | Accuracy | F1 | ROC-AUC | Recall |
|-------|-----------|----|----------|--------|
| Logistic Regression | 82.3 % | 0.64 | 0.84 | 0.68 |
| Random Forest | 86.5 % | 0.70 | 0.88 | 0.72 |
| Gradient Boosting | **88.1 %** | **0.72** | **0.90** | **0.75** |

The ensemble consistently outperforms single models by ~3 – 4 %, demonstrating better generalization.

<p align="center">
  <img src="./reports/voting.JPG" alt="Voting Ensemble" width="820">
</p>

**Figure 3 — Ensemble outcome:** balanced precision–recall and reduced variance.

---

## 🧠 Key Visual Insights

<p align="center">
  <img src="./reports/clustering_rules.JPG" alt="Cluster Visualization" width="800">
</p>

**Customer Clusters:** distinct churn behavior across demographics and contract types.

<p align="center">
  <img src="./reports/accc.JPG" alt="Model Accuracy Comparison" width="700">
</p>

**Model Performance Summary:** ensemble leads with 88 % accuracy.

<p align="center">
  <img src="./reports/workflow.jpg" alt="Workflow Overview" width="830">
</p>

**Overall Workflow:** unified flow from preprocessing to explainable predictions.

---

## ⚙️ How to Run

# 1. Clone this repository
git clone https://github.com/mwasifanwar/ChurnVision-AI.git
cd ChurnVision-AI

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # (Windows: .venv\Scripts\activate)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook

Execute notebooks in order:

Data_Preprocessing.ipynb

Clustering_Association.ipynb

Model_Training.ipynb

Voting_Ensemble.ipynb

Visualization_Report.ipynb

🚀 Future Enhancements

Integrate SHAP/LIME for transparent feature explanations.

Develop a FastAPI microservice for real-time predictions.

Add a Streamlit dashboard for business insights.

Automate retraining via MLOps (Docker + GitHub Actions).

Expand to cross-industry churn prediction datasets.

---

<br>

<h2 align="center">✨ Author</h2>

<p align="center">
  <b>Muhammad Wasif</b><br>
 AI/ML Developer • Founder @ Effixly AI
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/mwasifanwar" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn">
  </a>
  <a href="mailto:wasifsdk@gmail.com">
    <img src="https://img.shields.io/badge/Email-grey?style=for-the-badge&logo=gmail" alt="Email">
  </a>
  <a href="https://mwasif.dev" target="_blank">
    <img src="https://img.shields.io/badge/Website-black?style=for-the-badge&logo=google-chrome" alt="Website">
  </a>
</p>

<p align="center">
  <em>"Predicting churn isn’t just about saving customers — it’s about understanding them."</em>  
</p>

<br>

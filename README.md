# News Article Share Prediction

## 📰 Project Overview

This repository contains a machine learning project focused on **predicting the shareability of online news articles**.

The goal of this project is to analyze various features associated with news content, metadata, and publishing frequency to determine whether an article is likely to be highly shared (or "viral") on social media platforms. The primary output is a trained classification model that can assist content creators and publishers in understanding the factors driving audience engagement.

The entire analysis, from data ingestion and cleaning to model training and evaluation, is contained within the single Jupyter Notebook: `News_Article_Share_Prediction.ipynb`.

---

## 🛠️ Technology Stack

This project is built entirely in Python and relies on the standard data science and machine learning ecosystem.

* **Language:** Python 3.x

* **Data Analysis:** `pandas`, `numpy`

* **Machine Learning:** `scikit-learn` (for models, splitting, and metrics)

* **Visualization:** `matplotlib`, `seaborn`

* **Environment:** Jupyter Notebook / JupyterLab

---

## 🚀 Getting Started

Follow these instructions to set up your environment and run the analysis.

### Prerequisites

1. **Python:** Ensure you have Python 3.x installed.

2. **Jupyter:** Install Jupyter Notebook or JupyterLab.

   ```bash
   pip install jupyter


3. Installation and Setup

a. Clone the Repository:
   ```bash
   git clone [https://github.com/HisamSufian/news_article_prediction.git](https://github.com/HisamSufian/news_article_prediction.git)
   cd news_article_prediction
```

b. Install Dependencies: Install all required libraries using pip.
   ```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

# Running the Notebook 
1.  Launch Jupyter: jupyter notebook # OR jupyter lab

2.  Open the File: Navigate to and open News_Article_Share_Prediction.ipynb.

3.  Run Cells: Execute the notebook cells sequentially to perform the data loading, preprocessing, model training, and evaluation steps.

## 🧠 Methodology and Key Steps
The analysis within the notebook follows a typical machine learning workflow:

1.  Data Loading & Initial Exploration:
- Data is loaded from a xlsx.
- Initial checks for missing values, data types, and distribution of the target variable (share count/category).

2.  Feature Engineering & Preprocessing:
-Handling categorical features (e.g., one-hot encoding for publishing channels).
-Feature Scaling (Normalization or Standardization) to prepare numerical features for the model.
-Feature selection to identify the most impactful features (e.g., using correlation matrices or feature importance).

3.  Model Training (Classification):
- The problem is treated as a classification task (e.g., predicting 'High Share' or 'Low Share').
- Multiple algorithms are likely tested (e.g., Logistic Regression, Decision Trees, Random Forest, or XGBoost).
- Hyperparameter tuning (GridSearchCV or RandomizedSearchCV) is performed to optimize model performance.

4.  Evaluation:
- Models are evaluated using metrics such as Accuracy, Precision, Recall, F1-Score, and the ROC AUC curve.

## 📂 File Structure
news_article_prediction/
├── News_Article_Share_Prediction.ipynb        <- Main analysis notebook
└── README.md                                  <- This file
└── news_share_data.xlsx                       <- Assumed dataset file (may be internal to the repo)

## 📝 Future Enhancements
# Potential improvements for this project include:
. Deep Learning Models: Experimenting with Neural Networks (e.g., using TensorFlow or PyTorch) for potentially better prediction accuracy.

. Time Series Analysis: Incorporating time-based features to see if publishing trends influence shareability.

. Deployment: Creating a simple web application (using Flask or Streamlit) to host the trained model for real-time predictions.




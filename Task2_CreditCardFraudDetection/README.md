# Task 2: Credit Card Fraud Detection - CodSoft Internship

This project is part of the CodSoft Machine Learning Internship (Month Year, e.g., July 2024). The goal of this task is to build and evaluate machine learning models to detect fraudulent credit card transactions.

## Objective

To develop a classifier capable of identifying potentially fraudulent transactions based on historical transaction data, paying close attention to the challenges posed by imbalanced datasets.

## Dataset

The dataset was provided by CodSoft for this task and consists of two files:

- `fraudTrain.csv`: Used for training the models.
- `fraudTest.csv`: Used for evaluating the final models.

You can download the dataset directly from https://www.kaggle.com/datasets/kartik2112/fraud-detection (Link provided by CodSoft for Task 2).

**Note:** Due to the size of the dataset (it's quite large!), the `.csv` files are not included in this repository. Please download them using the link above and place them in this directory

## Methodology

1.  **Data Loading & Exploration:** Loaded the training and testing datasets using Pandas. Performed initial exploratory data analysis (EDA), checked for missing values, duplicates, and analyzed feature types. A key finding was the **severe class imbalance**, with fraudulent transactions representing less than 1% of the data.
2.  **Feature Engineering:**
    - Extracted useful features from `trans_date_trans_time` (hour, day of week, month).
    - Calculated customer `age` from `dob` and the transaction time.
    - Dropped unnecessary or potentially problematic columns (e.g., `Unnamed: 0`, `cc_num`, names, `trans_num`, high-cardinality categoricals like `merchant`, `city`, `job`).
3.  **Preprocessing:**
    - Applied **StandardScaler** to numerical features.
    - Applied **OneHotEncoder** to selected categorical features (`gender`, `category`, `state`).
    - Used `ColumnTransformer` to apply these steps consistently.
4.  **Handling Class Imbalance:** Employed **SMOTE (Synthetic Minority Over-sampling Technique)** on the _training data_ to create a balanced dataset for model training.
5.  **Model Training:** Trained three different classification models on the SMOTE-resampled data:
    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier
6.  **Model Evaluation:** Evaluated the trained models on the _original, processed test set_. Focused on metrics suitable for imbalanced classification:
    - Classification Report (Precision, Recall, F1-Score - especially for the fraud class)
    - Confusion Matrix
    - ROC AUC Score
    - Precision-Recall AUC (PR AUC) / Average Precision Score

## Results Summary

| Model               | Training Time (s) | ROC AUC    | PR AUC     | Avg Precision | Fraud Recall | Fraud Precision | Fraud F1-Score |
| ------------------- | ----------------- | ---------- | ---------- | ------------- | ------------ | --------------- | -------------- |
| Logistic Regression | ~83s              | 0.9022     | 0.1176     | 0.1177        | 0.73         | 0.03            | 0.06           |
| Decision Tree       | ~332s             | 0.8968     | 0.7146     | 0.5043        | 0.80         | 0.63            | 0.70           |
| **Random Forest**   | **~790s**         | **0.9837** | **0.8431** | **0.8394**    | **0.73**     | **0.91**        | **0.81**       |

**Conclusion:** The **Random Forest Classifier** demonstrated the best overall performance. While its recall for fraud (0.73) was comparable to Logistic Regression and slightly lower than the Decision Tree, its precision (0.91) was significantly higher, leading to the best F1-score (0.81) and PR AUC (0.84). This indicates it effectively identifies fraud while minimizing false positives, despite requiring the longest training time. The Decision Tree offered a good balance with high recall (0.80), while Logistic Regression suffered from extremely low precision.

## Libraries Used

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Imbalanced-learn

_(See `requirements.txt` for specific versions)_

## How to Run

1.  Ensure you have Python installed.
2.  Install the required libraries: `pip install -r requirements.txt`
3.  Download the `fraudTrain.csv` and `fraudTest.csv` files and place them in this directory.
4.  Run the Jupyter Notebook: `Task2_CreditCardFraudDetection.ipynb`

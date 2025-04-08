# Task 3: Customer Churn Prediction - CodSoft Internship

This project was completed as part of the CodSoft Machine Learning Internship (April 2025). The goal of this task is to build and evaluate machine learning models to predict customer churn for a bank, using historical customer data.

## Objective

To develop a classification model that can accurately identify bank customers who are likely to stop using the bank's services (churn). This allows the bank to implement targeted retention strategies for at-risk customers.

## Dataset

The dataset used for this project is the widely known "Churn Modelling" dataset, commonly found on Kaggle. It was provided or referenced by CodSoft for Task 3.

- **Filename:** `Churn_Modelling.csv`
- **Description:** Contains historical data about bank customers, including demographics, account details, and whether they exited the bank (churned).
- **Source Link:** https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction

**Note:** If the `.csv` file is not included in this repository, it can be downloaded from the source link above or obtained directly from CodSoft resources. Please place it in the project's root directory or relevant task folder for the code to run correctly.

## Methodology

The project followed these key steps:

1.  **Data Loading & Exploratory Data Analysis (EDA):**

    - Loaded the `Churn_Modelling.csv` dataset using Pandas.
    - Performed initial exploration: checked data shapes, column types (`df.info()`), missing values (`df.isnull().sum()`), and duplicate entries (`df.duplicated().sum()`). Found no missing values or duplicates.
    - Examined summary statistics (`df.describe()`).

2.  **Feature Engineering & Selection:**

    - Dropped columns deemed irrelevant or non-predictive for the churn model: `RowNumber`, `CustomerId`, `Surname`.

3.  **Data Preprocessing:**

    - **Categorical Features:** Applied One-Hot Encoding to `Geography` and `Gender` using `pd.get_dummies(..., drop_first=True)` to avoid multicollinearity.
    - **Train-Test Split:** Divided the data into training (80%) and testing (20%) sets using `train_test_split`, ensuring stratification based on the target variable `Exited` to maintain class proportions in both sets.
    - **Numerical Features:** Applied `StandardScaler` to scale numerical features (`CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`). Scaling was fitted _only_ on the training data and then applied to both training and testing sets to prevent data leakage.

4.  **Handling Class Imbalance (Partial):**

    - While explicit techniques like SMOTE were not used across all models, the Random Forest model was trained using `class_weight='balanced'` to give more importance to the minority class (churned customers). Stratification during the train-test split also helps ensure representation.

5.  **Model Training:**

    - Trained three distinct classification algorithms on the scaled training data:
      - Logistic Regression (`LogisticRegression`)
      - Random Forest Classifier (`RandomForestClassifier` with `class_weight='balanced'`)
      - Gradient Boosting Classifier (`GradientBoostingClassifier`)

6.  **Model Evaluation:**

    - Evaluated the trained models on the scaled _test set_.
    - Used the following metrics:
      - **Accuracy Score:** Overall correct predictions.
      - **ROC AUC Score:** Area Under the Receiver Operating Characteristic Curve, measuring discriminative ability.
      - **Confusion Matrix:** Visualizing TP, FP, TN, FN.
      - **Classification Report:** Providing precision, recall, and F1-score for both classes (Non-Churned: 0, Churned: 1).

7.  **Feature Importance Analysis:**
    - Extracted and ranked feature importances from the trained Gradient Boosting model to understand which factors most influence churn prediction.

## Results Summary

The performance of the models on the test set is summarized below:

| Model                 | Accuracy   | ROC AUC    | Churn Recall (Class 1) | Churn Precision (Class 1) | Churn F1-Score (Class 1) |
| :-------------------- | :--------- | :--------- | :--------------------- | :------------------------ | :----------------------- |
| Logistic Regression   | 0.8080     | 0.7748     | 0.19                   | 0.59                      | 0.28                     |
| Random Forest (Bal)   | 0.8610     | 0.8508     | 0.44                   | 0.78                      | 0.56                     |
| **Gradient Boosting** | **0.8700** | **0.8708** | **0.49**               | **0.79**                  | **0.60**                 |

**Key Feature Importances (from Gradient Boosting):**

1.  Age (0.388)
2.  NumOfProducts (0.300)
3.  IsActiveMember (0.114)
4.  Balance (0.089)
5.  Geography*Germany (0.056)
    *(Others had lower importance)\_

**Conclusion:**

The **Gradient Boosting Classifier** achieved the best overall performance among the tested models.

- It yielded the highest Accuracy, ROC AUC score, and F1-Score for the target churn class (1).
- While its churn recall (0.49) is moderate, it strikes a good balance with high precision (0.79), meaning when it predicts churn, it's often correct.
- The Random Forest model also performed well, benefiting from the `class_weight='balanced'` parameter, but Gradient Boosting edged it out slightly on most key metrics for churn prediction.
- Logistic Regression significantly underperformed in identifying churned customers (low recall and F1-score).
- Feature importance highlights that `Age`, `NumOfProducts`, `IsActiveMember`, and `Balance` are the most influential factors in predicting churn for this dataset.

Based on these results, the Gradient Boosting model is recommended for predicting customer churn in this scenario.

## Libraries Used

- Pandas
- NumPy
- Scikit-learn (`sklearn`)

_(See requirements.txt if provided for specific versions.)_

## How to Run

1.  **Clone the repository (optional):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>/Task_3_Customer_Churn
    ```
2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn jupyter
    ```
4.  **Obtain the dataset:** Download `Churn_Modelling.csv` from the [source link](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction) (or CodSoft) and ensure it's present in the correct directory.
5.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook Task3_CustomerChurnPrediction.ipynb
    ```
    (Or open and run the notebook using your preferred IDE.)

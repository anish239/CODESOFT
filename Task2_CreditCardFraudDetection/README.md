# Task 2: Credit Card Fraud Detection - CodSoft Internship

This project was completed as part of the CodSoft Machine Learning Internship (July 2024). The goal of this task is to build and evaluate machine learning models capable of detecting fraudulent credit card transactions using a real-world dataset.

## Objective

To develop a robust classification model that can accurately identify potentially fraudulent credit card transactions from historical data, with a specific focus on addressing the challenges presented by highly imbalanced datasets.

## Dataset

The dataset used for this project was provided by CodSoft and consists of transaction data. It is available for download from Kaggle:

- **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (Link provided by CodSoft for Task 2)
- **Files:**
  - `fraudTrain.csv`: Used for training the models.
  - `fraudTest.csv`: Used for evaluating the final models.

**Note:** Due to the large size of the dataset, the `.csv` files are not included in this repository. Please download them from the link above and place them in the project's root directory.

## Methodology

The project followed these key steps:

1.  **Data Loading & Exploratory Data Analysis (EDA):**

    - Loaded the training (`fraudTrain.csv`) and testing (`fraudTest.csv`) datasets using Pandas.
    - Performed initial exploration: checked data shapes, column types, missing values, and duplicate entries.
    - Identified **severe class imbalance**: Fraudulent transactions constitute less than 1% of the data, which poses a significant challenge for model training and evaluation.

2.  **Feature Engineering:**

    - Extracted temporal features (`hour`, `day_of_week`, `month`) from the `trans_date_trans_time` column.
    - Calculated customer `age` based on their `dob` (date of birth) and the transaction timestamp.
    - Dropped columns deemed unnecessary or potentially problematic for modeling:
      - Identifiers: `Unnamed: 0`, `cc_num`, `trans_num`
      - Personal Info: `first`, `last`, `street`, `zip`
      - High Cardinality Categorical: `merchant`, `city`, `job` (due to complexity and potential overfitting)
      - Raw Date/Time: `trans_date_trans_time`, `dob` (after feature extraction)

3.  **Data Preprocessing:**

    - **Numerical Features:** Applied `StandardScaler` to scale features like `amt`, `city_pop`, `lat`, `long`, `merch_lat`, `merch_long`, `age`, `hour`, `day_of_week`, `month`.
    - **Categorical Features:** Applied `OneHotEncoder` (handle_unknown='ignore') to selected low-to-medium cardinality features: `gender`, `category`, `state`.
    - Used `ColumnTransformer` to apply these transformations consistently to both training and testing datasets.

4.  **Handling Class Imbalance:**

    - Applied **SMOTE (Synthetic Minority Over-sampling Technique)** from the `imbalanced-learn` library.
    - **Important:** SMOTE was applied _only_ to the preprocessed _training data_ to avoid data leakage into the test set. This creates a balanced dataset for model training.

5.  **Model Training:**

    - Trained three distinct classification algorithms on the SMOTE-resampled training data:
      - Logistic Regression
      - Decision Tree Classifier
      - Random Forest Classifier

6.  **Model Evaluation:**
    - Evaluated the trained models on the _original, preprocessed test set_ (i.e., without SMOTE applied).
    - Focused on metrics suitable for imbalanced classification tasks:
      - **Confusion Matrix:** To visualize true positives, false positives, true negatives, and false negatives.
      - **Classification Report:** Providing precision, recall, and F1-score, particularly focusing on the minority (fraud) class.
      - **ROC AUC Score:** Area Under the Receiver Operating Characteristic Curve.
      - **Precision-Recall AUC (PR AUC) / Average Precision Score:** A more informative metric than ROC AUC for highly imbalanced datasets.

## Results Summary

The performance of the models on the test set is summarized below:

| Model               | Training Time (s) | ROC AUC    | PR AUC (Avg Precision) | Fraud Recall | Fraud Precision | Fraud F1-Score |
| :------------------ | :---------------- | :--------- | :--------------------- | :----------- | :-------------- | :------------- |
| Logistic Regression | ~83               | 0.9022     | 0.1177                 | 0.73         | 0.03            | 0.06           |
| Decision Tree       | ~332              | 0.8968     | 0.5043                 | 0.80         | 0.63            | 0.70           |
| **Random Forest**   | **~790**          | **0.9837** | **0.8394**             | **0.73**     | **0.91**        | **0.81**       |

_(Note: Training times are approximate and depend on the execution environment.)_
_(Note: PR AUC and Avg Precision are often used interchangeably; the value reported here is the Average Precision Score calculated by scikit-learn, which summarizes the PR curve.)_

**Conclusion:**

The **Random Forest Classifier** achieved the best overall performance, particularly in terms of balancing precision and recall for the minority fraud class.

- While its fraud recall (0.73) was similar to Logistic Regression and slightly lower than the Decision Tree (0.80), its fraud precision (0.91) was significantly superior.
- This resulted in the highest Fraud F1-Score (0.81) and the highest Precision-Recall AUC (0.84), indicating it's the most effective model at identifying fraud cases while minimizing the number of legitimate transactions flagged as fraudulent (false positives).
- The Decision Tree showed good recall but moderate precision.
- Logistic Regression struggled significantly with precision, making it less practical despite reasonable recall.
- The trade-off for Random Forest's superior performance is its longer training time.

Based on these results, the Random Forest model is recommended for this fraud detection task.

## Libraries Used

- Pandas
- NumPy
- Scikit-learn (`sklearn`)
- Imbalanced-learn (`imblearn`)
- Matplotlib
- Seaborn

_(See `requirements.txt` for specific versions used.)_

## How to Run

1.  **Clone the repository (optional):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the dataset:** Obtain `fraudTrain.csv` and `fraudTest.csv` from the [Kaggle link](https://www.kaggle.com/datasets/kartik2112/fraud-detection) and place them in the project's root directory.
5.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook Task2_CreditCardFraudDetection.ipynb
    ```
    (Or open and run the notebook using your preferred IDE like VS Code, PyCharm, etc.)

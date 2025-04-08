# Task 4: Spam SMS Detection - CodSoft Internship

This project was completed as part of the CodSoft Machine Learning Internship (April 2025). The goal of this task is to build and evaluate machine learning models to classify SMS messages as either "spam" or "ham" (not spam).

## Objective

To develop and compare different classification models capable of accurately detecting spam SMS messages using a standard dataset, focusing on text preprocessing and feature extraction techniques suitable for text data.

## Dataset

The dataset used for this project is the "SMS Spam Collection Dataset".

- **Source:** (https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **File:** `spam.csv` (The dataset might need to be downloaded and potentially renamed or preprocessed slightly from the original source to match the `spam.csv` structure used in the notebook, which commonly has 'v1' and 'v2' columns initially).
- **Encoding:** The file is loaded using `latin-1` encoding due to potential special characters often found in this dataset.
- **Content:** Contains two main columns: a label ('ham' or 'spam') and the raw text message.

**Note:** The `spam.csv` file needs to be downloaded from the source (or a similar source like Kaggle where it's often hosted) and placed in the same directory as the notebook.

## Methodology

The project followed these key steps:

1.  **Data Loading & Cleaning:**

    - Loaded the `spam.csv` dataset using Pandas with `latin-1` encoding.
    - Removed extraneous 'Unnamed' columns sometimes present in CSV formats derived from this dataset.
    - Renamed the default columns ('v1', 'v2') to more descriptive names ('label', 'message').
    - Checked for and handled missing values (none found in this typical version).
    - Identified and removed duplicate entries to ensure data integrity.

2.  **Text Preprocessing:**

    - Implemented a `preprocess_text` function to clean the SMS messages:
      - Converted text to lowercase.
      - Removed punctuation using `string.punctuation`.
      - Removed digits using regular expressions (`re`).
      - Tokenized the text into individual words using `nltk.word_tokenize`.
      - Removed common English stop words using `nltk.corpus.stopwords`.
      - Filtered out very short words (length <= 2).
      - Rejoined the processed words into a clean string.
    - Applied this function to the `message` column, creating a new `processed_message` column.

3.  **Feature Engineering (Vectorization):**

    - Converted the cleaned text data (`processed_message`) into numerical features using `TfidfVectorizer` from Scikit-learn.
      - Limited the vocabulary size to the top 3000 features (`max_features=3000`).
      - Included both unigrams and bigrams (`ngram_range=(1, 2)`) to capture more context.
    - Created a numerical target variable `label_num` by mapping 'ham' to 0 and 'spam' to 1.

4.  **Data Splitting:**

    - Split the TF-IDF feature matrix (`X`) and the numerical labels (`y`) into training (80%) and testing (20%) sets using `train_test_split`.
    - Used `random_state=42` for reproducibility.
    - Applied `stratify=y` to ensure the proportion of spam/ham messages was similar in both the training and testing sets, which is important for reliable evaluation.

5.  **Model Training:**

    - Trained three common classification algorithms suitable for text data on the vectorized training data:
      - Multinomial Naive Bayes (`MultinomialNB`)
      - Logistic Regression (`LogisticRegression`, with increased `max_iter`)
      - Support Vector Machine (`SVC`, with `probability=True`)

6.  **Model Evaluation:**
    - Evaluated the trained models on the held-out **test set**.
    - Used standard classification metrics, focusing on performance for the minority 'spam' class (label=1):
      - **Accuracy:** Overall correctness.
      - **Precision (Spam):** Of messages predicted as spam, how many actually were spam.
      - **Recall (Spam):** Of all actual spam messages, how many were correctly identified.
      - **F1-Score (Spam):** The harmonic mean of precision and recall for spam.
      - **Confusion Matrix:** To visualize true/false positives and negatives.
      - **Classification Report:** Detailed precision, recall, F1-score for both classes.

## Results Summary

The performance of the models on the test set is summarized below:

| Model               | Accuracy   | Precision (Spam) | Recall (Spam) | F1-Score (Spam) |
| :------------------ | :--------- | :--------------- | :------------ | :-------------- |
| Naive Bayes         | 0.9710     | 0.9903           | 0.7786        | 0.8718          |
| Logistic Regression | 0.9478     | 0.9639           | 0.6107        | 0.7477          |
| **SVM**             | **0.9710** | **0.9810**       | **0.7863**    | **0.8729**      |

**Conclusion:**

Both **Multinomial Naive Bayes** and **Support Vector Machine (SVM)** performed exceptionally well and very similarly on this task, achieving ~97.1% accuracy.

- They both demonstrate high precision, meaning the messages they flagged as spam were highly likely to actually be spam.
- SVM showed a slightly better recall (0.79 vs 0.78 for NB) and F1-Score (0.873 vs 0.872 for NB), indicating a marginally better ability to identify _all_ spam messages present in the test set.
- Logistic Regression performed well but lagged behind NB and SVM, particularly in recall for the spam class (0.61).
- Given their near-identical high performance, either Naive Bayes (often faster and simpler) or SVM could be chosen. SVM has a very slight edge in the balanced F1-score for spam detection in this specific evaluation.

Based on these results, SVM is marginally the best-performing model, closely followed by Naive Bayes.

## Libraries Used

- Pandas
- NLTK (Natural Language Toolkit)
- Scikit-learn (`sklearn`)
- re (Regular Expressions)
- string

## How to Run

1.  **Clone or download the repository/code:**
    ```bash
    # If using git
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install pandas nltk scikit-learn notebook
    # You might also need to run Python and download NLTK data:
    # python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
    ```
    (Ideally, use a `requirements.txt` file if provided: `pip install -r requirements.txt`)
4.  **Download the dataset:** Obtain the dataset from the (https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (or a similar source) and ensure it's saved as `spam.csv` in the project's root directory. You might need to adjust the downloaded file to match the format expected by the notebook (e.g., ensuring columns are named 'v1'/'v2' or directly 'label'/'message', and using 'latin-1' encoding if necessary).
5.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook Task4_SpamSmsDetection.ipynb
    ```
    (Or open and run the notebook using your preferred IDE like VS Code, PyCharm, JupyterLab, etc.)

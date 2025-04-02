# CODSOFT Internship - Task 1: Movie Genre Classification

## Description

This repository contains the Jupyter Notebook (`Task1_Movie_Genre_Classification.ipynb`) for Task 1 of the CodSoft Machine Learning Internship. The goal of this task was to build and evaluate machine learning models (specifically Multinomial Naive Bayes, Logistic Regression, and Linear SVM) to predict a movie's genre based on its plot summary, using TF-IDF features for text representation.

## Data

The dataset files (`train_data.txt`, `test_data.txt`, `test_data_solution.txt`) used for this project are large (~35MB each) and therefore **are not included** in this repository due to GitHub file size recommendations.

**➡️ Please download the required dataset files from:**
[(https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)]

## Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YourGitHubUsername/CODSOFT.git
    cd CODSOFT
    ```

    _(Replace the URL with your actual repository URL)_

2.  **Download the data:** Obtain the `train_data.txt`, `test_data.txt`, and `test_data_solution.txt` files from the link provided in the **Data** section above.

3.  **Place data files:** Move the downloaded `.txt` files into the main directory of this cloned repository (the same folder where the `.ipynb` file is).

4.  **Install Libraries:** Ensure you have the necessary Python libraries installed.

    ```bash
    pip install pandas numpy nltk scikit-learn
    ```

5.  **Download NLTK Data:** Run Python and execute the following commands if you haven't already:

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

6.  **Run the Notebook:** Open and run the cells in the `Task1_Movie_Genre_Classification.ipynb` Jupyter Notebook.

## Results Summary

The models were trained on the training data and evaluated on the test data, yielding the following accuracies:

- **Logistic Regression Accuracy:** 57.95%
- **Linear SVM Accuracy:** 57.11%
- **Multinomial Naive Bayes Accuracy:** 52.24%

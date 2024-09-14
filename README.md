# Job Advertisement Classification using Natural Language Processing

## Overview

This project focuses on building an automated job advertisement classification system using Natural Language Processing (NLP). The tasks include pre-processing a collection of job advertisements, generating different feature representations, and building machine learning models to classify the job advertisements into their respective categories.

The project is divided into three main tasks:

1. **Task 1: Basic Text Preprocessing**
2. **Task 2: Generating Feature Representations**
3. **Task 3: Job Advertisement Classification**

## Files Included

- **count_vectors.txt**: Stores the sparse count vector representation of job advertisement descriptions.
- **data.zip**: Contains the dataset of job advertisements divided into multiple categories such as Accounting, Finance, Engineering, etc.
- **stopwords_en.txt**: A list of stopwords used for text preprocessing.
- **vocab.txt**: Contains the vocabulary built from the preprocessed job descriptions.
- **task1.ipynb**: Jupyter notebook implementing Task 1 (Basic Text Preprocessing).
- **task2_3.ipynb**: Jupyter notebook implementing Task 2 (Feature Representation) and Task 3 (Job Advertisement Classification).

## Tasks Summary

### Task 1: Basic Text Preprocessing

- **Goal**: Preprocess the text data from the job advertisements.
- **Steps**:
  1. Tokenization
  2. Lowercasing all words
  3. Removal of words shorter than 2 characters
  4. Stopword removal (using `stopwords_en.txt`)
  5. Removal of words appearing only once
  6. Removal of the top 50 most frequent words
- **Output**: 
  - A cleaned vocabulary saved in `vocab.txt`
  - Preprocessed job descriptions saved in an appropriate format for further tasks.

### Task 2: Generating Feature Representations

- **Goal**: Generate different feature representations for the job advertisements.
- **Methods**:
  - **Count Vectors**: Sparse count vector representation based on the vocabulary generated in Task 1.
  - **Word Embeddings**: Representation of job descriptions using pre-trained word embedding models like Word2Vec, FastText, or GloVe.
  - **TF-IDF Weighting**: Apply TF-IDF weighting to the word embeddings.
- **Output**: 
  - `count_vectors.txt`: Contains sparse count vector representation of each job advertisement description.

### Task 3: Job Advertisement Classification

- **Goal**: Build machine learning models to classify job advertisements into categories.
- **Experiments**:
  - **Q1: Language Model Comparisons**: Compare the performance of different feature representations (Count Vectors, TF-IDF, Word Embeddings) for classifying job advertisements.
  - **Q2: Adding More Information**: Investigate whether adding more information, such as the job title, improves the accuracy of classification.
- **Methods**:
  - Machine learning models such as **Logistic Regression**.
  - **5-fold Cross-Validation** is used for model evaluation.
  
## How to Run

1. **Environment Setup**:
   - Ensure you have Python 3.x installed.
   - Install the necessary libraries:
     ```bash
     pip install pandas numpy scikit-learn matplotlib
     ```

2. **Run Task 1**:
   - Open `task1.ipynb` and execute all cells to preprocess the job advertisement data.
   - The vocabulary will be saved as `vocab.txt`.

3. **Run Tasks 2 and 3**:
   - Open `task2_3.ipynb` and execute the cells to generate feature representations and build machine learning models for job classification.
   - Results will be saved in `count_vectors.txt` and displayed within the notebook.

## Outputs

- **Vocabulary File**: `vocab.txt`
- **Sparse Count Vectors**: `count_vectors.txt`

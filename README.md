# Fake-News-Detection-System
Fake News Detection System using Natural Language Processing and Logistic Regression. This project classifies news articles as real or fake based on textual content using TF-IDF vectorization and logistic regression. Built with Python, NLTK, and scikit-learn on the large ISOT Kaggle dataset.

# Fake News Detection System

A machine learning project that detects whether a news article is real or fake using Natural Language Processing (NLP) and Logistic Regression classification.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Interactive Prediction](#interactive-prediction)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Project Overview
Fake news can cause misinformation and disrupt social order. This project builds a robust automated system to classify news articles as real or fake based on their text content. Using NLP techniques like text cleaning and TF-IDF vectorization together with a Logistic Regression model, it achieves high accuracy on a large, balanced dataset.

The project is implemented in Python using libraries like Pandas, NLTK, and scikit-learn. It includes an interactive terminal mode to test predictions on any input news text.

## Dataset
This project uses the **ISOT Fake News Dataset** obtained from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).  
- Over 44,000 news articles  
- Includes two CSV files:
  - `True.csv` (real news)  
  - `Fake.csv` (fake news)  
- Balanced binary classification labels: `0` for real news and `1` for fake news

## Features
- Data preprocessing: text cleaning (lowercase, punctuation removal, stopword removal)  
- TF-IDF vectorization with top 5,000 tokens  
- Logistic Regression classifier  
- Stratified train-test split ensuring balanced data distribution  
- Interactive prediction mode for live testing  

## Installation
1. Clone this repository:
   (https://github.com/dhruv7noxa/Fake-News-Detection-System)

2. pip install pandas scikit-learn nltk


3. Download the dataset files `Fake.csv` and `True.csv` from Kaggle and place them in the project folder.

## Usage

Run the main script to train the model and use the interactive prediction mode:


The script will:

- Load and combine the dataset  
- Preprocess the text  
- Train a Logistic Regression model  
- Evaluate the model with accuracy and classification metrics  
- Enter an interactive mode allowing users to input news text and get real/fake predictions

Type `exit` to quit the interactive mode.

## Model Details

- **Training Data:** ~36,000 samples (80%)  
- **Testing Data:** ~9,000 samples (20%)  
- **Algorithm:** Logistic Regression (`max_iter=200`)  
- **Feature Engineering:** TF-IDF vectorization (max 5,000 features)  

## Results

- **Accuracy:** Approximately 98.7%  
- **Precision, Recall, F1-score:** ~99% for both classes  
- **Confusion Matrix:** Very low misclassification rate in test data  

## Interactive Prediction

After training, the script allows users to type any news headline or article and get an instant prediction:

Enter a news article text (or type 'exit' to quit): NASA confirms water on Mars
🟢 Prediction: REAL news


## Future Work

- Integrate advanced NLP models like BERT or LSTM for deeper understanding  
- Deploy as a web or mobile application using Flask or Streamlit  
- Support multilingual fake news detection  
- Include metadata and source reliability for better predictions  

## Acknowledgements

- ISOT Fake News Dataset: [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
- Python libraries: Pandas, scikit-learn, NLTK  
- Tutorials and official documentation from scikit-learn and NLTK  

## License

This project is licensed under the MIT License.

---

Feel free to customize the GitHub URL and your name/email in the description accordingly. If you want, I can also help you prepare a Python requirements file or a detailed CONTRIBUTING guide for the repo.

Would you like me to generate the complete `fake_news.py` with a saved model option and instructions on how to run that interactive mode from the README?



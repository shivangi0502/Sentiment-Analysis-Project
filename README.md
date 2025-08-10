# Sentiment Analysis Project

A Python project for multi-class sentiment analysis on Twitter data. It features a pipeline for text processing and ML model training (LR, RF, LinearSVC), a Hugging Face transformer pipeline, and a Streamlit chatbot for interactive testing and model insights.

## Overview

This project demonstrates a complete end-to-end pipeline for building a multi-class sentiment analysis application. The workflow covers data preprocessing, feature engineering with TF-IDF, training of traditional machine learning models, and integration of a powerful pre-trained Hugging Face transformer model. The final application is a user-friendly chatbot built with Streamlit.

## Features

- **Multi-class Sentiment Analysis**: Classifies text into three categories: Negative, Neutral, and Positive.
- **Data Preprocessing**: Includes robust text cleaning, tokenization, lemmatization, and stopword removal.
- **Multiple Models**: Compares the performance of Logistic Regression, Random Forest, Linear Support Vector Classifier (LinearSVC), and a Hugging Face transformer model.
- **Interactive Chatbot**: A Streamlit web application that allows users to input text and get real-time sentiment predictions.
- **Model Insights**: Provides a dedicated section in the app to visualize model performance metrics such as confusion matrices, ROC curves, and F1-scores.
- **Word Cloud Visualization**: Generates a WordCloud from the user's input to highlight key terms.

## Technologies Used

- **Python**: Core programming language.
- **scikit-learn**: For traditional machine learning models and evaluation metrics.
- **Hugging Face `transformers`**: For utilizing a pre-trained sentiment analysis model (`cardiffnlp/twitter-roberta-base-sentiment`).
- **NLTK**: For text preprocessing tasks like tokenization and lemmatization.
- **Streamlit**: For building the interactive web application.
- **Pandas, NumPy, Matplotlib, Seaborn**: For data manipulation, analysis, and visualization.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/sentiment_analysis_project.git](https://github.com/your-username/sentiment_analysis_project.git)
    cd sentiment_analysis_project
    ```
2.  **Create and activate a Conda environment:**
    ```bash
    conda create -n sentiment_env python=3.9 -y
    conda activate sentiment_env
    ```
3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    (Note: You may need to install PyTorch or TensorFlow separately based on your system for the Hugging Face model.)
4.  **Download NLTK data:**
    In a Python interpreter or a Jupyter Notebook, run:
    ```python
    import nltk
    nltk.download('all')
    ```

## Usage

1.  **Run the Jupyter Notebooks in order:**
    - `01_data_exploration_and_cleaning.ipynb`
    - `02_feature_engineering_and_modeling.ipynb`
    - `03_huggingface_pipeline.ipynb`
    This will process the data, train the models, and save all necessary artifacts (models, vectorizers, evaluation data) to the `models/` directory.

2.  **Launch the Streamlit app:**
    Make sure your `sentiment_env` is active and run:
    ```bash
    streamlit run app.py
    ```

The app will open in your browser, and you can start testing the models and exploring their insights.

import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.preprocessor import TextPreprocessor
from transformers import pipeline
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'], ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    return fig

#function to generate a multi-class ROC curve plot
def plot_roc_curve(y_true, y_proba, labels):
    fig, ax = plt.subplots(figsize=(8, 6))
    class_names = ['Negative', 'Neutral', 'Positive']
    
    for i, class_label in enumerate(labels):
        y_true_binary = (y_true == class_label).astype(int)
        y_proba_class = y_proba[:, i]
        fpr, tpr, _ = roc_curve(y_true_binary, y_proba_class)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'ROC for {class_names[i]} (area = {roc_auc:.2f})')
    
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    return fig

# Load all resources (models, vectorizer, evaluation data, etc.)
@st.cache_resource
def load_resources():
    project_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_root, 'models')
    data_dir = os.path.join(project_root, 'data')

    preprocessor = TextPreprocessor()
    
    # Load TF-IDF vectorizer
    with open(os.path.join(models_dir, 'tfidf_vectorizer_multiclass.pkl'), 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    
    # Load all traditional ML models
    all_models = {} 
    ml_model_names = ['Logistic Regression', 'Random Forest', 'Linear SVC']
    for name in ml_model_names:
        file_name = name.lower().replace(' ', '_') + '_model_multiclass.pkl'
        model_path = os.path.join(models_dir, file_name)
        with open(model_path, 'rb') as f:
            all_models[name] = pickle.load(f)

    # Load Hugging Face model 
    hf_model_name = 'cardiffnlp/twitter-roberta-base-sentiment'
    hf_classifier = pipeline('sentiment-analysis', model=hf_model_name)
    all_models['Hugging Face'] = hf_classifier 

    
    with open(os.path.join(models_dir, 'y_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    eval_data = {}
    for model_name in ml_model_names: 
        file_name = model_name.lower().replace(' ', '_') + '_eval_data.pkl'
        with open(os.path.join(models_dir, file_name), 'rb') as f:
            eval_data[model_name] = pickle.load(f)
    with open(os.path.join(models_dir, 'hf_eval_data.pkl'), 'rb') as f:
        hf_eval_data_from_file = pickle.load(f)
    eval_data['Hugging Face'] = hf_eval_data_from_file

    # Load dataset for summary
    df_raw = pd.read_csv(os.path.join(data_dir, 'tweet_eval_sentiment_multiclass_preprocessed.csv'))
    
    return preprocessor, tfidf_vectorizer, all_models, y_test, eval_data, df_raw

preprocessor, tfidf_vectorizer, all_models, y_test, eval_data, df_raw = load_resources()

# Map sentiment labels 
sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
hf_label_to_display_map = {'LABEL_0': "Negative", 'LABEL_1': "Neutral", 'LABEL_2': "Positive"}
class_labels = [0, 1, 2]

#Streamlit App UI
st.set_page_config(page_title="Multi-class Sentiment Chatbot", layout="wide")
st.title("Sentiment Analysis Chatbot")

#Tabs for Chatbot and Model Insights
tab1, tab2 = st.tabs(["Chatbot", "Model Insights"])

with tab1:
    st.write("Analyze the sentiment of a message using different models.")
    
    st.header("Chatbot")
    with st.sidebar:
        st.header("Model Selector")
        selected_model_name = st.selectbox(
            "Choose a Model:",
            options=list(all_models.keys()), 
            index=3
        )
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("What is the sentiment of your message?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            st.write(f"Analyzing sentiment with **{selected_model_name}**...")
            
            #Conditional Prediction Logic
            if selected_model_name == 'Hugging Face':
                hf_result = all_models['Hugging Face'](prompt)[0]
                hf_label = hf_result['label']
                hf_score = hf_result['score']
                hf_display_sentiment = hf_label_to_display_map.get(hf_label, "Unknown")
                st.write(f"**Hugging Face Prediction:** {hf_display_sentiment} (Confidence: {hf_score:.2f})")
            else:
                ml_model = all_models[selected_model_name]
                processed_input = preprocessor.preprocess(prompt)
                input_vectorized = tfidf_vectorizer.transform([processed_input])
                ml_prediction = ml_model.predict(input_vectorized)[0]
                ml_sentiment = sentiment_map[ml_prediction]
                st.write(f"**{selected_model_name} Prediction:** {ml_sentiment}")
            
            with st.expander("Show Word Cloud"):
                processed_input = preprocessor.preprocess(prompt) # Re-process for WordCloud
                if processed_input.strip():
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(processed_input)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info("No words to generate a Word Cloud from this message.")

# In app.py
with tab2:
    st.header("Model Insights")
    st.markdown("Here you can explore the performance of the trained models on the test dataset.")
    
    # Dataset Summary Section
    with st.expander("Dataset Summary"):
        st.write(f"Total samples in original dataset: {len(df_raw)}")
        st.write("Sentiment Distribution:")
        st.bar_chart(df_raw['sentiment'].value_counts())
        st.write("Sample data:")
        st.dataframe(df_raw.sample(5, random_state=42))

    # Insights for each model
    for model_name, data in eval_data.items():
        with st.expander(f"**{model_name} Insights**"):
            y_pred = data['y_pred']
            
            
            y_true_for_eval = data.get('y_true', y_test)
            y_proba = data.get('y_proba')
            
            acc = accuracy_score(y_true_for_eval, y_pred)
            prec = precision_score(y_true_for_eval, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_true_for_eval, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true_for_eval, y_pred, average='weighted', zero_division=0)
            
            st.write("### Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{acc:.2f}")
            col2.metric("Precision", f"{prec:.2f}")
            col3.metric("Recall", f"{rec:.2f}")
            col4.metric("F1-Score", f"{f1:.2f}")
            
            st.write("### Plots")
            col1, col2 = st.columns(2)
            with col1:
                st.write("#### Confusion Matrix")
                fig_cm = plot_confusion_matrix(y_true_for_eval, y_pred, labels=class_labels)
                st.pyplot(fig_cm)
            
            with col2:
                
                if y_proba is not None and y_proba.ndim > 1 and y_proba.shape[1] == len(class_labels):
                    st.write("#### ROC Curve (One-vs-Rest)")
                    fig_roc = plot_roc_curve(y_true_for_eval, y_proba, labels=class_labels)
                    st.pyplot(fig_roc)
                else:
                    st.info("ROC Curve not available for this model's output or full data.")
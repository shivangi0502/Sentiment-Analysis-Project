from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
import pickle

class TextVectorizer:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.word2vec_model = None
        
    def fit_transform_tfidf(self,corpus):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        X_tfidf = self.tfidf_vectorizer.fit_transform(corpus)
        return X_tfidf
    
    def transform_tfidf(self,corpus):
        if self.tfidf_vectorizer is None:
            raise ValueError("TF -IDF vectorizer not fitted yet.")
        return self.tfidf_vectorizer.transform(corpus)
    
    def train_word2vec(self, tokenized_sentences, vector_size =100, window= 5, min_count = 1, workers =4):
        self.word2vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=vector_size,
                                       window=window, min_count=min_count, workers=workers)
        self.word2vec_model.train(tokenized_sentences, total_examples=len(tokenized_sentences), epochs=10)
        
    def get_word_embedding_vector(self, word):
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model not trained yet. Call train_word2vec first.")
        try:
            return self.word2vec_model.wv[word]
        except KeyError:
            return np.zeros(self.word2vec_model.vector_size) # Return zeros for out-of-vocabulary words
        
    def get_sentence_embedding(self, tokenized_sentence):
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model not trained yet. Call train_word2vec first.")
        vectors = [self.get_word_embedding_vector(word) for word in tokenized_sentence if word in self.word2vec_model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.word2vec_model.vector_size)

    def save_tfidf_vectorizer(self, path='models/tfidf_vectorizer.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)

    def load_tfidf_vectorizer(self, path='models/tfidf_vectorizer.pkl'):
        with open(path, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)

    def save_word2vec_model(self, path='models/word2vec_model.bin'):
        self.word2vec_model.save(path)

    def load_word2vec_model(self, path='models/word2vec_model.bin'):
        self.word2vec_model = Word2Vec.load(path)

        
     
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC , LinearSVC
from sklearn.model_selection import train_test_split
import pickle

class ModelTrainer:
    def __init__(self):
        self.model = None
        
    
    def train_model(self, X, y, model_type='logistic_regression', **kwargs):
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, **kwargs)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=42, **kwargs)
        elif model_type == 'svm':
            self.model = SVC(probability=True, random_state=42, **kwargs) 
        elif model_type == 'linear_svc':
            self.model = LinearSVC(random_state=42, **kwargs)
        else:
            raise ValueError("Unsupported model type. Choose from 'logistic_regression', 'random_forest', 'svm'.")

        self.model.fit(X, y)
        print(f"{model_type.replace('_', ' ').title()} model trained successfully.")
        
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            return self.model.decision_function(X)
        else:
            raise AttributeError("Model does not support predict_proba.")

    def save_model(self, path='models/trained_model.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")
        
    def load_model(self, path ='models/trained_model.pkl'):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"model loaded from {path}")
        
if __name__ =='__main__':
    from sklearn.datasets import make_classification
    X,y= make_classification(n_samples=100, n_features=10,random_state=42)
    X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.3,random_state=42)
    
    trainer = ModelTrainer()
    trainer.train_model(X_train,y_train, model_type='logistic_regression')
    predictions = trainer.predict(X_test)
    print(f"sample predictions: {predictions[:5]}")
    trainer.save_model()
    
    new_trainer = ModelTrainer()
    new_trainer.load_model()
    new_predictions = new_trainer.predict(X_test)
    print(f"new predictions (after loading): {new_predictions[:5]}")
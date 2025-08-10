from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ModelEvaluator:
    def __init__(self):
        pass

    def evaluate(self, y_true, y_pred, y_proba=None, model_name="Model"):
        print(f"\n--- Evaluation for {model_name} ---")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
        print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}")
        print(f"F1-Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

        # Determine class labels from the data
        class_labels = sorted(np.unique(np.concatenate((y_true, y_pred))))
        sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        target_names = [sentiment_labels[label] for label in class_labels]

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix for {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

        # ROC Curve (for multi-class, one-vs-rest)
        if y_proba is not None and y_proba.shape[1] > 1:
            plt.figure(figsize=(10, 8))
            for i, class_label in enumerate(class_labels):
                y_true_binary = (y_true == class_label).astype(int)
                y_proba_class = y_proba[:, i]
                fpr, tpr, _ = roc_curve(y_true_binary, y_proba_class)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'ROC curve for {target_names[i]} (area = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Receiver Operating Characteristic (ROC) Curve for {model_name}')
            plt.legend(loc="lower right")
            plt.show()

if __name__ == '__main__':
    # Example usage for testing
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1])
    y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.6, 0.4], [0.7, 0.3],
                        [0.1, 0.9], [0.3, 0.7], [0.4, 0.6], [0.95, 0.05], [0.05, 0.95]])

    evaluator = ModelEvaluator()
    evaluator.evaluate(y_true, y_pred, y_proba, model_name="Test Model")
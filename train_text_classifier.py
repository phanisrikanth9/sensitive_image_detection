import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # Import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
from pattern_detector import PatternDetector
from sklearn.calibration import CalibratedClassifierCV
import joblib
from config import Config

# Custom transformer for pattern detection
class PatternDetector(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.patterns = Config.SENSITIVE_PATTERNS
    
    def transform(self, X):
        features = []
        for text in X:
            counts = [len(re.findall(pattern, text)) for pattern in self.patterns.values()]
            features.append(counts)
        return np.array(features)
    
    def fit(self, X, y=None):
        return self

# Text classification model
class TextClassifier:
    def __init__(self):
        self.pipeline = self._build_pipeline()
    def _build_pipeline(self):
        return Pipeline([
            ('features', FeatureUnion([
                ('tfidf', TfidfVectorizer(
                    stop_words='english',
                    max_features=500,  # Reduce complexity
                    ngram_range=(1,2)  # Add bigrams
                )),
                ('patterns', PatternDetector())
            ])),
            ('classifier', LogisticRegression(
                class_weight='balanced',
                C=0.1,  # Increase regularization
                max_iter=1000
            ))
        ])
    def train(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Optimize threshold for high precision
        probas = self.pipeline.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, probas)
        
        # Find threshold with minimum 95% precision
        threshold = next(t for p, t in zip(precision, thresholds) if p >= 0.95)
        print(f"Optimal threshold: {threshold:.2f}")
        return threshold

def main():
    # Load and combine datasets
    dfs = [pd.read_csv(f) for f in Config.SYNTHETIC_FILES + 
           [Config.SENSITIVE_FILE, Config.NON_SENSITIVE_FILE]]
    data = pd.concat(dfs).sample(frac=1, random_state=Config.RANDOM_STATE)
    
    # Convert labels to numeric
    data['label'] = data['label'].map({'Sensitive': 1, 'Non-Sensitive': 0})
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'],
        test_size=Config.TEST_SIZE,
        stratify=data['label'],
        random_state=Config.RANDOM_STATE
    )
    
    # Train and evaluate the model
    classifier = TextClassifier()
    classifier.train(X_train, y_train)
    threshold = classifier.evaluate(X_test, y_test)
    
    # Save the trained model
    joblib.dump({
        'model': classifier.pipeline,
        'threshold': threshold,
        'config': {
            'SENSITIVE_PATTERNS': Config.SENSITIVE_PATTERNS,
            'MODEL_SAVE_PATH': Config.MODEL_SAVE_PATH
        }
    }, Config.MODEL_SAVE_PATH)
    print(f"Model saved to {Config.MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()
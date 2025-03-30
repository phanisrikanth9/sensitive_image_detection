# pattern_detector.py
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from config import Config

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
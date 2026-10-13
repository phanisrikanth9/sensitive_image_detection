"""scikit-learn transformer that turns text into validated PII counts.

Lives in its own importable module (instead of being re-defined inside the
training script) so the saved pipeline can be loaded anywhere with joblib -
this removes the old `sys.modules['__main__'].PatternDetector = ...` hack.
"""
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from detectors import PII_TYPES, pii_counts

# Context words that often surround sensitive data even when no ID number is
# visible (e.g. a bank statement header, a medical report, a password note).
KEYWORD_GROUPS = {
    "secrecy": r"\b(confidential|secret|restricted|private|internal use only|do not share|classified)\b",
    "credentials": r"\b(password|passwd|username|login|otp|pin|cvv|api key|token)\b",
    "identity": r"\b(aadhaar|aadhar|passport|license|licence|ssn|social security|pan card|date of birth|dob|uid)\b",
    "financial": r"\b(account|a/c|ifsc|routing|bank|salary|payslip|credit|debit|balance|card)\b",
    "medical": r"\b(diagnosis|patient|prescription|medical|hospital|blood group|lab report|mrn)\b",
}
_KEYWORD_RES = {k: re.compile(v, re.I) for k, v in KEYWORD_GROUPS.items()}

FEATURE_NAMES = [f"pii_{t}" for t in PII_TYPES] + [f"kw_{k}" for k in KEYWORD_GROUPS]


class PatternDetector(BaseEstimator, TransformerMixin):
    """Features = [count of each validated PII type] + [count of each keyword group]."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            text = text or ""
            kw = [len(r.findall(text)) for r in _KEYWORD_RES.values()]
            rows.append(pii_counts(text) + kw)
        # log1p keeps one huge count from dominating the linear model
        return np.log1p(np.array(rows, dtype=float))

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES)

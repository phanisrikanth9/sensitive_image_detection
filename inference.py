import joblib
import re
from pattern_detector import PatternDetector
from config import Config

class SensitivityChecker:
    def __init__(self, model_path):
        # Load the model and its components
        model_data = joblib.load(model_path)
        self.pipeline = model_data['model']
        self.threshold = model_data.get('threshold', 0.5)  # Use default threshold if not found
        self.patterns = Config.SENSITIVE_PATTERNS

    def validate_ssn(self, ssn):
        """Validate SSN against official rules"""
        ssn = re.sub(r'[ -]', '', ssn)  # Remove separators
        if len(ssn) != 9: return False
        if ssn.startswith(('000', '666')) or ssn[0] == '9':
            return False
        return True

    def validate_phone(self, phone):
        """Validate Indian phone numbers"""
        digits = re.sub(r'[^0-9]', '', phone)
        return len(digits) == 12 and digits.startswith(('6', '7', '8', '9'))

    def validate_credit_card(self, card):
        """Validate credit card using Luhn algorithm"""
        card = re.sub(r'[ -]', '', card)
        if len(card) != 16 or not card.isdigit():
            return False
        # Luhn algorithm
        total = 0
        for i, digit in enumerate(reversed(card)):
            num = int(digit)
            if i % 2 == 1:
                num *= 2
                if num > 9:
                    num = num - 9
            total += num
        return total % 10 == 0

    def contains_sensitive_pattern(self, text):
        """Enhanced validation for all patterns"""
        for name, pattern in self.patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                value = match.group()
                if name == 'ssn' and not self.validate_ssn(value):
                    continue
                if name == 'phone' and not self.validate_phone(value):
                    continue
                if name == 'credit_card' and not self.validate_credit_card(value):
                    continue
                return True
        return False

    def predict(self, text):
        """Predict whether the text is sensitive."""
        # Rule-based check first
        if self.contains_sensitive_pattern(text):
            return 1
        
        # Model prediction with probability validation
        try:
            proba = self.pipeline.predict_proba([text])[0][1]
            return 1 if proba >= self.threshold else 0
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0  # Fallback to non-sensitive

if __name__ == '__main__':
    checker = SensitivityChecker(Config.MODEL_SAVE_PATH)
    
    test_cases = [
        "My credit card is 4111-1111-1111-1111",  # Valid Visa (Sensitive)
        "Meeting at 3 PM in room 205",  # Non-sensitive
        "Software version 2.3.45",  # Safe number (Non-sensitive)
        "SSN: 123-45-6789",  # Invalid SSN (Non-sensitive)
        "Phone: +91-1234567890",  # Invalid phone (Non-sensitive)
        "Thank you for your application, Phani Srikanth! We look forward to reviewing it! Browse More Jobs",  # Non-sensitive
        "Lets meet at 8pm",  # Non-sensitive
        "Yoo 123 Yoo",  # Non-sensitive
        "397788000234",  # Random numbers (Non-sensitive)
        "Aadhaar: 2345 6789 0123"  # Valid Aadhaar (Sensitive)
    ]
    
    for text in test_cases:
        prediction = checker.predict(text)
        print(f"Text: {text}\nPrediction: {'Sensitive' if prediction else 'Non-Sensitive'}\n")
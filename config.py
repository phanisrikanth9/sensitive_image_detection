# config.py
class Config:
    # Dataset parameters
    NUM_SYNTHETIC_SAMPLES = 500
    NUM_SENSITIVE_SAMPLES = 1000
    NUM_NON_SENSITIVE_SAMPLES = 1000
    SENSITIVE_RATIO = 0.5  # 50-50 balance
    
    # File paths
    SYNTHETIC_FILES = [
        'synthetic_dataset1.csv',
        'synthetic_dataset2.csv',
        'synthetic_dataset3.csv'
    ]
    SENSITIVE_FILE = 'sensitive_dataset.csv'
    NON_SENSITIVE_FILE = 'non_sensitive_dataset.csv'
    
    # Model parameters
    TEST_SIZE = 0.25
    RANDOM_STATE = 42
    MODEL_SAVE_PATH = 'text_classification_pipeline.pkl'
    
    # Pattern configurations
    SENSITIVE_PATTERNS = {
        'credit_card': r'\b(?:\d[ -]*?){13,16}\b',  # Matches credit card numbers with/without spaces/dashes
        'ssn': r'\b\d{3}[ -]?\d{2}[ -]?\d{4}\b',    # Matches SSNs with/without spaces/dashes
        'phone': r'\b\+?\d[ -]?(?:\d[ -]?){9,14}\b', # Matches phone numbers with/without spaces/
        'aadhaar': r'\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b',  # UIDAI format
        'atm_pin': r'\b\d{4}\b',
        'pan': r'\b[A-Z]{5}\d{4}[A-Z]\b'
    }
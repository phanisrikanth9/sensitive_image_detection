import random
import csv
import string
from config import Config

class NonSensitiveDataGenerator:
    def __init__(self):
        self.base_corpora = [
            [
                "The quarterly financial review indicates stable growth across all market sectors",
                "Technical documentation updates should be finalized by next Thursday",
                "Literary analysis of postmodern works requires understanding of historical context"
            ],
            [
                "Software version updates will be deployed in phases starting next week",
                "Team building exercises are scheduled for Friday afternoon",
                "The new coding standards document has been approved by the committee"
            ]
        ]
    
    def safe_number_generator(self):
        """Generate numbers that cannot be mistaken for sensitive patterns"""
        formats = [
            lambda: f"{random.randint(10, 31)}/{random.randint(1, 12)}",  # Dates
            lambda: f"v{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 99)}",  # Versions
            lambda: f"Room {random.choice(['A','B','C'])}{random.randint(100, 999)}",  # Locations
            lambda: f"{random.randint(1, 9)}.{random.randint(1, 9)}%",  # Percentages
        ]
        return random.choice(formats)()

    def generate_sample(self):
        """Generate realistic non-sensitive text samples"""
        base_text = random.choice(random.choice(self.base_corpora))
        
        # Add safe numerical data with 40% probability
        if random.random() < 0.4:
            numerical_data = self.safe_number_generator()
            insertion_point = random.randint(0, len(base_text.split()))
            words = base_text.split()
            words.insert(insertion_point, numerical_data)
            base_text = ' '.join(words)
            
        return [base_text, "Non-Sensitive"]

    def generate_dataset(self):
        """Generate and save dataset with proper shuffling"""
        dataset = [self.generate_sample() for _ in range(Config.NUM_NON_SENSITIVE_SAMPLES)]
        random.shuffle(dataset)
        
        with open(Config.NON_SENSITIVE_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            writer.writerows(dataset)
        print(f"Generated {len(dataset)} non-sensitive samples")

if __name__ == '__main__':
    generator = NonSensitiveDataGenerator()
    generator.generate_dataset()
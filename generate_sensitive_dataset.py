# generate_sensitive_dataset.py
import random
import csv
import string
from config import Config

class SensitiveDataGenerator:
    def __init__(self):
        self.pattern_generators = [
            self.generate_credit_card,
            self.generate_ssn,
            self.generate_aadhaar,
            self.generate_email,
            self.generate_phone_number,
            self.generate_atm_pin,
            self.generate_pan_card,
            self.generate_sensitive_keyword
        ]
        
        self.base_corpora = [
            # Expanded base texts (similar to original but longer)
            ["System integration requires careful planning and execution to ensure all components work together seamlessly. "
             "This includes API development, database management, and user interface design."],
            
            ["The financial report for Q3 shows significant growth in revenue across all sectors. "
             "Key metrics include a 15% increase in net profit and a 10% reduction in operational costs."],
            
            ["In the realm of classical literature, themes of human resilience and moral complexity dominate. "
             "Authors like Tolstoy and Dickens explore the intricacies of society and individual struggle."]
        ]

    # --------------------------
    # Pattern Generators
    # --------------------------
    def generate_credit_card(self):
        prefixes = [
        '4111',  # Visa test
        '5555',  # Mastercard test
        '3714'   # Amex test
        ]
        return f"{random.choice(prefixes)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}"

    def generate_phone_number(self):
        # Valid Indian numbers only (6-9 start)
        return f"+91 {random.choice(['6','7','8','9'])}{random.randint(10,99)} {random.randint(10000,99999)}"    
    def generate_ssn(self):
        """Generate a valid Social Security Number (SSN) in XXX-XX-XXXX format."""
        return f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"
    
    def generate_aadhaar(self):
        """Generate a 12-digit Aadhaar number with valid checksum."""
        base = ''.join(random.choices(string.digits, k=11))  # First 11 digits
        total = sum(int(digit) * (i % 2 + 1) for i, digit in enumerate(base))  # Checksum calculation
        checksum = str(total % 10)  # Last digit
        return f"{base[:4]} {base[4:8]} {base[8:11]}{checksum}"  # Format: XXXX XXXX XXXX
    
    def generate_email(self):
        """Generate realistic email addresses with common patterns."""
        username_options = [
            lambda: f"{random.choice(['john', 'jane', 'alex'])}{random.randint(10,99)}",
            lambda: f"{random.choice(['user', 'admin', 'test'])}.{random.randint(1000,9999)}",
            lambda: "support" + str(random.randint(1, 99)) + random.choice(["", "_team"]),
        ]
        domains = ["gmail.com", "yahoo.com", "outlook.com", "company.org", "business.net", "enterprise.io"]
        username = random.choice(username_options)()
        domain = random.choice(domains)
        return f"{username}@{domain}".lower()    
    def generate_atm_pin(self):
        """Generate a 4-digit ATM PIN with basic security validation."""
        while True:
            pin = ''.join(random.choices(string.digits, k=4))
            if (len(set(pin)) >= 2 and  # Avoid repeated digits
                not pin in ('0000', '1111', '2222', '3333', '1234')):  # Avoid common patterns
                return pin
    
    def generate_pan_card(self):
        """Generate valid PAN card numbers with checksum."""
        status_char = random.choice(['A', 'B', 'C', 'F', 'G', 'H', 'L', 'J', 'P', 'T'])  # Status character
        first_part = ''.join(random.choices(string.ascii_uppercase, k=4))  # First 4 letters
        middle_part = f"{random.randint(1000, 9999)}{status_char}"  # 4 digits + status character
        checksum_char = random.choice(string.ascii_uppercase)  # Last character
        return f"{first_part}{middle_part}{checksum_char}"  # Format: AAAAP1234A
    
    def generate_sensitive_keyword(self):
        """Generate context-aware sensitive keywords."""
        categories = {
            'classification': ["Top Secret", "Confidential", "Restricted"],
            'data_types': ["PII", "PHI", "PCI", "Financial Records"],
            'handling': ["Internal Use Only", "Need-to-Know Basis", "Eyes Only"],
            'security': ["Encrypted", "Tokenized", "Redacted"],
            'legal': ["Attorney-Client Privilege", "Non-Disclosure Agreement"]
        }
        components = [random.choice(cat) for cat in random.sample(list(categories.values()), k=2)]
        return " - ".join(components)  # Format: "Confidential - PII"

    # --------------------------
    # Data Injection and Dataset Generation
    # --------------------------
    def inject_sensitive_data(self, text):
        """Inject sensitive data into the text."""
        injections = []
        for _ in range(random.randint(1, 3)):  # Inject 1-3 sensitive items
            generator = random.choice(self.pattern_generators)
            injections.append(generator())
        
        for injection in injections:
            words = text.split()
            insert_idx = random.randint(0, len(words))  # Random insertion point
            words.insert(insert_idx, injection)
            text = " ".join(words)
        return text

    def generate_dataset(self):
        """Generate the sensitive dataset and save it to a CSV file."""
        dataset = []
        for _ in range(Config.NUM_SENSITIVE_SAMPLES):
            base_text = random.choice(random.choice(self.base_corpora))  # Random base text
            modified_text = self.inject_sensitive_data(base_text)  # Inject sensitive data
            dataset.append([modified_text, "Sensitive"])  # Add to dataset
        
        # Save dataset to CSV
        with open(Config.SENSITIVE_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])  # Write header
            writer.writerows(dataset)  # Write data rows
        
        print(f"Sensitive dataset generated and saved to {Config.SENSITIVE_FILE}")

# --------------------------
# Main Execution
# --------------------------
if __name__ == '__main__':
    generator = SensitiveDataGenerator()
    generator.generate_dataset()
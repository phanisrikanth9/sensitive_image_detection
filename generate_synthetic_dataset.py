# generate_synthetic_dataset.py
import random
import csv
import string
from config import Config

def generate_random_text():
    """Generate random text for synthetic data."""
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "System integration requires careful planning.",
        "Financial reports show significant growth.",
        "In the realm of classical literature, themes of human resilience dominate.",
        "Cloud computing and big data analytics drive business decisions."
    ]
    return random.choice(sentences)

def generate_synthetic_dataset(output_file, num_samples, sensitive_ratio=0.5):
    """Generate a synthetic dataset with sensitive and non-sensitive samples."""
    dataset = []
    for _ in range(num_samples):
        text = generate_random_text()
        if random.random() < sensitive_ratio:
            # Inject sensitive data
            sensitive_data = [
                f"Credit Card: {random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}",
                f"SSN: {random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}",
                f"Phone: +91-{random.choice(['6','7','8','9'])}{random.randint(100000000, 999999999)}"
            ]
            text += " " + random.choice(sensitive_data)
            label = "Sensitive"
        else:
            label = "Non-Sensitive"
        dataset.append([text, label])
    
    # Save to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(dataset)
    print(f"Synthetic dataset saved to {output_file}")

if __name__ == '__main__':
    # Generate 3 synthetic datasets
    generate_synthetic_dataset("synthetic_dataset1.csv", num_samples=100)
    generate_synthetic_dataset("synthetic_dataset2.csv", num_samples=100)
    generate_synthetic_dataset("synthetic_dataset3.csv", num_samples=100)
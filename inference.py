"""Command-line checker for text or images.

    python inference.py --text "My card is 4111 1111 1111 1111"
    python inference.py --image path/to/photo.jpg
    python inference.py            # runs the built-in text examples
"""
import argparse
import json

from analyzer import PrivacyAnalyzer

EXAMPLES = [
    ("My credit card is 4111-1111-1111-1111", True),   # Luhn-valid Visa test number
    ("Meeting at 3 PM in room 205", False),
    ("Software version 2.3.45", False),
    ("SSN: 123-45-6789", True),                         # valid SSN format/rules
    ("Phone: +91-1234567890", False),                   # invalid number; ML still flags the "Phone:" label (known false alarm)
    ("Call me on +91 98765 43210", True),
    ("Thank you for your application! Browse More Jobs", False),
    ("Happy New Year 2025", False),                     # old system: flagged as ATM PIN
    ("397788000234", False),                            # bare 12 digits, no Aadhaar label
    ("Aadhaar: 2345 6789 0123", True),                  # labelled Aadhaar
    ("my gmail password is sunshine", True),            # no ID number: caught by ML model
    ("Patient diagnosed with tuberculosis", True),
]


def check_text(analyzer, text):
    findings, score = analyzer.analyze_text(text)
    thr = analyzer.text_model.threshold if analyzer.text_model else None
    sensitive = bool(findings) or (score is not None and score >= thr)
    return sensitive, findings, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--image")
    args = ap.parse_args()

    if args.image:
        analyzer = PrivacyAnalyzer()
        with open(args.image, "rb") as f:
            res = analyzer.analyze_image(f.read(), args.image, redact_output=False)
        print(json.dumps(res, indent=2))
        return

    analyzer = PrivacyAnalyzer(load_image_model=False, load_ocr=False)
    for w in analyzer.warnings:
        print("warning:", w)
    cases = [(args.text, None)] if args.text else EXAMPLES
    correct = 0
    for text, expected in cases:
        sensitive, findings, score = check_text(analyzer, text)
        why = ", ".join(f"{f.type} {f.masked}" for f in findings) or (f"ML score {score:.2f}" if score is not None else "-")
        mark = "" if expected is None else ("  OK" if sensitive == expected else "  WRONG")
        correct += expected is not None and sensitive == expected
        print(f"{'SENSITIVE ' if sensitive else 'safe      '} | {why:<32} | {text}{mark}")
    if not args.text:
        print(f"\n{correct}/{len(cases)} examples correct")


if __name__ == "__main__":
    main()

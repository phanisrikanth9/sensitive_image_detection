"""Generate the synthetic text dataset used to train the text classifier.

Why it was rewritten
--------------------
The old generators inserted IDs into only 3 (sensitive) and 6 (non-sensitive)
fixed sentences. The model could reach 100% accuracy just by memorising which
sentence it was looking at, and 65% of the non-sensitive rows were exact
duplicates, so the same sentence appeared in both train and test.

This version:
  * uses ~80 different sentence *templates* written to look like real OCR output
    (ID cards, bank statements, payslips, chats, posters, menus, slides, tickets...)
  * puts realistic, checksum-valid fake IDs into the sensitive templates
  * deliberately adds "hard negatives": harmless text full of numbers
    (years, prices, order numbers, flight times, masked card numbers...)
  * includes sensitive text with NO ID number at all (passwords, diagnoses,
    "confidential" memos) so the ML model has something to learn beyond the regex
  * simulates OCR mistakes (O<->0, l<->1, merged words, ALL CAPS)
  * records which template each row came from, so training can split by
    template and test on wordings the model has never seen

All personal data is fake: card numbers use test prefixes, names are random.
"""
import csv
import os
import random
import string

from config import Config
from detectors import luhn_valid, verhoeff_check_digit

R = random.Random(Config.RANDOM_STATE)

FIRST = ["Rahul", "Priya", "Anil", "Sneha", "Vikram", "Lakshmi", "John", "Emily", "Carlos", "Mei",
         "Arjun", "Divya", "Kiran", "Sara", "David", "Fatima", "Ravi", "Anjali", "Michael", "Aisha"]
LAST = ["Sharma", "Reddy", "Rao", "Patel", "Kumar", "Iyer", "Smith", "Garcia", "Chen", "Khan",
        "Naidu", "Gupta", "Brown", "Wilson", "Das", "Varma"]
CITY = ["Hyderabad", "Tempe", "Phoenix", "Bengaluru", "Chennai", "Vijayawada", "Austin", "Mumbai", "Pune"]
COMPANY = ["Acme Corp", "Northwind", "Globex", "Initech", "Stark Labs", "Umbrella Inc", "Wayne Enterprises"]
BANK = ["State Bank of India", "HDFC Bank", "ICICI Bank", "Chase", "Wells Fargo", "Axis Bank"]
DIAG = ["Type 2 diabetes", "hypertension", "major depressive disorder", "asthma", "HIV positive", "fractured radius"]
FOOD = ["Paneer Tikka", "Veg Biryani", "Margherita Pizza", "Cold Coffee", "Masala Dosa", "Caesar Salad"]
EVENT = ["Tech Fest", "Hackathon", "Career Fair", "Music Night", "Orientation", "Alumni Meet"]


def name():
    return f"{R.choice(FIRST)} {R.choice(LAST)}"


def digits(n):
    return "".join(R.choices(string.digits, k=n))


def card():
    prefix = R.choice(["4", "51", "52", "55", "37", "6011"])
    length = 15 if prefix == "37" else 16
    body = prefix + digits(length - len(prefix) - 1)
    check = next(d for d in "0123456789" if luhn_valid(body + d))
    n = body + check
    sep = R.choice([" ", "-", ""])
    return sep.join(n[i:i + 4] for i in range(0, len(n), 4))


def aadhaar():
    base = R.choice("23456789") + digits(10)
    n = base + verhoeff_check_digit(base)
    return f"{n[:4]} {n[4:8]} {n[8:]}"


def ssn():
    return f"{R.randint(100, 665):03d}-{R.randint(1, 99):02d}-{R.randint(1, 9999):04d}"


def pan():
    return "".join(R.choices(string.ascii_uppercase, k=3)) + R.choice("PCHF") + R.choice(string.ascii_uppercase) \
        + digits(4) + R.choice(string.ascii_uppercase)


def phone():
    if R.random() < 0.6:
        n = R.choice("6789") + digits(9)
        return R.choice([f"+91 {n[:5]} {n[5:]}", f"+91-{n}", n])
    return f"({R.randint(201, 989)}) {R.randint(201, 989)}-{digits(4)}"


def email():
    return f"{R.choice(FIRST).lower()}.{R.choice(LAST).lower()}{R.randint(1, 99)}@{R.choice(['gmail.com', 'asu.edu', 'outlook.com', 'yahoo.in'])}"


def account():
    return digits(R.choice([11, 12, 14]))


def ifsc():
    return R.choice(["SBIN", "HDFC", "ICIC", "UTIB"]) + "0" + digits(6)


def password():
    return "".join(R.choices(string.ascii_letters + string.digits + "!@#$", k=R.randint(8, 12)))


def date():
    return f"{R.randint(1, 28):02d}/{R.randint(1, 12):02d}/{R.randint(1975, 2006)}"


def recent_date():
    return f"{R.randint(1, 28):02d}-{R.randint(1, 12):02d}-{R.randint(2023, 2026)}"


def money():
    return f"{R.randint(1, 9999)}.{R.randint(0, 99):02d}"


SLOTS = {
    "name": name, "card": card, "aadhaar": aadhaar, "ssn": ssn, "pan": pan, "phone": phone,
    "email": email, "acct": account, "ifsc": ifsc, "pwd": password, "dob": date, "date": recent_date,
    "amt": money, "city": lambda: R.choice(CITY), "company": lambda: R.choice(COMPANY),
    "bank": lambda: R.choice(BANK), "diag": lambda: R.choice(DIAG), "food": lambda: R.choice(FOOD),
    "event": lambda: R.choice(EVENT), "year": lambda: str(R.randint(2019, 2027)),
    "n2": lambda: str(R.randint(10, 99)), "n3": lambda: str(R.randint(100, 999)),
    "n4": lambda: str(R.randint(1000, 9999)), "n5": lambda: str(R.randint(10000, 99999)),
    "otp": lambda: digits(R.choice([4, 6])), "last4": lambda: digits(4),
    "time": lambda: f"{R.randint(1, 12)}:{R.choice(['00', '15', '30', '45'])} {R.choice(['AM', 'PM'])}",
}

SENSITIVE = [
    # --- identity documents (OCR of ID cards) ---
    "GOVERNMENT OF INDIA {name} DOB: {dob} Male {aadhaar} Aadhaar - Aam Aadmi ka Adhikar",
    "Unique Identification Authority of India Enrolment No {n4}/{n5}/{n5} To {name} {city} Your Aadhaar No. {aadhaar}",
    "INCOME TAX DEPARTMENT GOVT. OF INDIA {name} Date of Birth {dob} Permanent Account Number {pan}",
    "Permanent Account Number Card {pan} Name {name} Father's Name {name} Signature",
    "Social Security {ssn} This number has been established for {name} SIGNATURE",
    "DRIVER LICENSE {city} DL {n4}{n5} DOB {dob} {name} EXP {date} CLASS D",
    "Name: {name} SSN: {ssn} Address: {n3} Main St {city}",
    "REPUBLIC OF INDIA PASSPORT Surname {name} Date of Birth {dob} Place of Issue {city}",
    # --- financial ---
    "{bank} VISA {card} VALID THRU {n2}/{n2} {name}",
    "Card Number {card} Expiry {n2}/{n2} CVV {n3} Cardholder {name}",
    "{bank} Account Statement A/c No {acct} IFSC {ifsc} Customer {name} Closing Balance {amt}",
    "Please transfer to A/c {acct} IFSC {ifsc} beneficiary {name} {bank}",
    "Payslip for the month {company} Employee {name} PAN {pan} Net Salary {amt} Bank A/c {acct}",
    "Debit card PIN: {otp} do not share with anyone",
    "Your OTP is {otp} for transaction of Rs {amt} at {company}. Do not share it with anyone",
    "Routing number {n4}{n5} Account number {acct} {bank} wire instructions",
    "Credit card bill {bank} card ending {card} total due {amt} pay by {date}",
    "W-2 Wage and Tax Statement Employee SSN {ssn} Employer {company} Wages {amt}",
    # --- credentials ---
    "WiFi password: {pwd}",
    "username {email} password {pwd}",
    "Login ID {name} Password {pwd} please change after first login",
    "API key {pwd}{pwd}{pwd} keep this secret",
    "My netbanking login is {email} and the password is {pwd}",
    "Admin credentials for the server user root pass {pwd}",
    # --- medical ---
    "Patient Name {name} DOB {dob} Diagnosis {diag} Prescribed medication twice daily",
    "{city} General Hospital Lab Report Patient {name} MRN {n5}{n3} Result: {diag}",
    "Discharge summary for {name} admitted with {diag} contact {phone}",
    "Medical record of {name} blood group B+ history of {diag}",
    # --- contact details / personal messages ---
    "Hey its {name}, my number is {phone} and email {email}",
    "Contact {name} at {phone} for the apartment keys",
    "Emergency contact {name} {phone} relationship father",
    "Send the documents to {email} and call me on {phone}",
    "Shipping to {name} {n3} Apartment {city} phone {phone}",
    # --- confidential business ---
    "CONFIDENTIAL - Internal use only. Q{n2} layoffs planned for {company} engineering team",
    "Strictly confidential: salary revision for {name} new CTC {amt} lakh",
    "Restricted - do not share. Merger talks between {company} and {company}",
    "Private and confidential offer letter {name} annual compensation {amt}",
    "Internal use only {company} customer list with phone numbers and emails",
    "Classified document distribution limited to need-to-know personnel",
    "Board minutes - confidential. Acquisition price {amt} million agreed",
]

NON_SENSITIVE = [
    # --- signs, posters, events ---
    "{event} {year} Join us on {date} at {time} Room {n3} Free entry",
    "Welcome to {city} Please keep the campus clean",
    "Grand Opening {company} new store {date} flat {n2}% off",
    "{event} registrations close {date} Prizes worth Rs {n5}",
    "No parking zone Tow away area Violators will be fined",
    "Exit Emergency use only Push bar to open",
    "Library hours Monday to Friday {time} to {time}",
    "Happy New Year {year} from all of us at {company}",
    "Lost cat answers to Whiskers reward offered",
    "Bus {n3} to {city} departs every {n2} minutes",
    # --- food / shopping ---
    "Menu {food} {n3} {food} {n3} {food} {n3} GST extra",
    "Thank you for shopping at {company} Order #{n5} Total {amt} paid by card XXXX XXXX XXXX {last4}",
    "Receipt {date} {food} x2 {amt} Subtotal {amt} Tax {amt} Total {amt}",
    "Invoice {year}-{n4} Qty {n2} Unit price {amt} Amount {amt}",
    "Buy 1 get 1 free on {food} this weekend only",
    "Your order {n5}{n3} will arrive by {date}",
    "Price {amt} MRP inclusive of all taxes Batch {n4} Mfg {date}",
    # --- travel ---
    "Flight AI {n3} {city} to {city} departs {time} Gate {n2} Seat {n2}A",
    "Train {n5} Coach S{n2} Platform {n2} arrival {time}",
    "Weather {city} {n2} degrees sunny humidity {n2}%",
    "{city} {n3} km Speed limit {n2}",
    # --- work / study ---
    "Lecture {n2} Introduction to Machine Learning gradient descent and backpropagation",
    "Agenda for {date}: sprint review at {time} then retro",
    "Assignment {n2} due {date} submit on Canvas",
    "Version {n2}.{n2}.{n3} release notes bug fixes and performance improvements",
    "Error code {n4} please restart the application",
    "Slide {n2} Results accuracy improved by {n2}% over baseline",
    "Meeting moved to {time} in conference room {n4}",
    "Quarterly revenue grew {n2}% year over year according to the public report",
    "Chapter {n2} The history of the Roman Empire",
    "Table {n2} Population of {city} in {year} was {n2} lakh",
    "Exam timetable Paper {n3} on {date} from {time}",
    # --- casual chat / social ---
    "Lets meet at {time} near the {city} mall",
    "Happy birthday {name}! Have a great year ahead",
    "lol that movie was amazing see you tomorrow",
    "Can you send me the notes for lecture {n2}",
    "Congratulations to the winners of {event} {year}",
    "Score {n2} - {n2} what a match",
    "Good morning have a nice day",
    "Steps today {n5} calories {n4}",
]


OCR_SWAPS = {"O": "0", "0": "O", "l": "1", "I": "1", "S": "5", "B": "8", "e": "c"}


def ocr_noise(text):
    """Simulate typical OCR mistakes (character swaps, merged words, caps)."""
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch in OCR_SWAPS and R.random() < 0.05:
            chars[i] = OCR_SWAPS[ch]
    text = "".join(chars)
    if R.random() < 0.3:
        words = text.split()
        if len(words) > 3:
            j = R.randrange(len(words) - 1)
            words[j:j + 2] = [words[j] + words[j + 1]]
        text = " ".join(words)
    if R.random() < 0.2:
        text = text.upper()
    return text


def fill(template):
    out = template
    while "{" in out:
        start = out.index("{")
        end = out.index("}", start)
        out = out[:start] + SLOTS[out[start + 1:end]]() + out[end + 1:]
    return out


def build():
    rows = []
    for label, templates, prefix in [("Sensitive", SENSITIVE, "S"), ("Non-Sensitive", NON_SENSITIVE, "N")]:
        for t_id, template in enumerate(templates):
            seen = set()
            for _ in range(Config.SAMPLES_PER_TEMPLATE * 3):
                text = fill(template)
                if R.random() < Config.OCR_NOISE_PROB:
                    text = ocr_noise(text)
                if text in seen:
                    continue
                seen.add(text)
                rows.append([text, label, f"{prefix}{t_id:02d}"])
                if len(seen) == Config.SAMPLES_PER_TEMPLATE:
                    break
    R.shuffle(rows)
    return rows


def main():
    rows = build()
    os.makedirs(os.path.dirname(Config.TEXT_DATASET_PATH), exist_ok=True)
    with open(Config.TEXT_DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "template"])
        w.writerows(rows)
    n_s = sum(r[1] == "Sensitive" for r in rows)
    print(f"Wrote {len(rows)} rows ({n_s} sensitive, {len(rows) - n_s} non-sensitive) "
          f"from {len(SENSITIVE)} + {len(NON_SENSITIVE)} templates -> {Config.TEXT_DATASET_PATH}")


if __name__ == "__main__":
    main()

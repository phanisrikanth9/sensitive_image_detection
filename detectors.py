"""Rule-based PII detection: regular expressions + checksum / format validation.

A regex alone only checks the *shape* of a string ("12 digits"), which produces
many false alarms (a year like 2024 looks like a 4-digit PIN). So every candidate
match is passed through a validator that applies the real rules for that ID type:

    credit card  -> Luhn checksum + known card-network prefixes
    Aadhaar      -> Verhoeff checksum (the algorithm UIDAI actually uses)
    SSN          -> SSA rules (no 000/666/9xx area, no 00 group, no 0000 serial)
    PAN          -> 4th character must be a valid holder-type code
    phone        -> Indian mobile (starts 6-9) or US/NANP number
    PIN/OTP/CVV  -> only counted when a keyword like "PIN:" sits right before it

Each finding comes back with its type, position and a masked value, so the app can
explain *why* an image was flagged and blur exactly that region.
"""
import re
from dataclasses import dataclass, asdict

# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------

def luhn_valid(number: str) -> bool:
    """Luhn (mod 10) checksum used by every major card network."""
    digits = [int(d) for d in number]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Verhoeff tables (dihedral group D5) - catches all single-digit errors and
# all adjacent transpositions, which is why UIDAI chose it for Aadhaar.
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_valid(number: str) -> bool:
    c = 0
    for i, d in enumerate(reversed(number)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(d)]]
    return c == 0


def verhoeff_check_digit(number: str) -> str:
    """Compute the digit to append so that verhoeff_valid() is True."""
    c = 0
    for i, d in enumerate(reversed(number)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][int(d)]]
    return str(_VERHOEFF_INV[c])


# ---------------------------------------------------------------------------
# Validators (receive the raw matched string)
# ---------------------------------------------------------------------------

def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def valid_credit_card(s: str) -> bool:
    d = _digits(s)
    if not 13 <= len(d) <= 19 or not luhn_valid(d):
        return False
    # Visa, Mastercard (51-55, 2221-2720), Amex, Discover, RuPay, JCB, Diners
    return bool(re.match(r"4|5[1-5]|2(2[2-9]|[3-6]\d|7[01]|720)|3[47]|6(011|5|0|52)|35|3[068]", d))


def valid_aadhaar(s: str) -> bool:
    d = _digits(s)
    return len(d) == 12 and d[0] not in "01" and verhoeff_valid(d)


def valid_ssn(s: str) -> bool:
    d = _digits(s)
    area, group, serial = d[:3], d[3:5], d[5:]
    return (len(d) == 9 and area not in ("000", "666") and area[0] != "9"
            and group != "00" and serial != "0000")


def valid_pan(s: str) -> bool:
    # 4th char = holder type: P person, C company, H HUF, F firm, A AOP, T trust,
    # B BOI, L local authority, J juridical person, G government
    return s[3] in "PCHFATBLJG"


def valid_phone(s: str) -> bool:
    d = _digits(s)
    if len(d) == 12 and d.startswith("91"):
        d = d[2:]
    elif len(d) == 11 and d[0] == "0":
        d = d[1:]
    if len(d) == 10 and d[0] in "6789" and len(set(d)) > 2:
        return True  # Indian mobile
    if len(d) == 11 and d[0] == "1":
        d = d[1:]
    # US / NANP: area code and exchange can't start with 0 or 1
    return len(d) == 10 and d[0] in "23456789" and d[3] in "23456789"


# ---------------------------------------------------------------------------
# Patterns. Order matters: when two matches overlap, the earlier type wins.
# ---------------------------------------------------------------------------

PATTERNS = {
    "email":       re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    "aadhaar":     re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b"),
    "pan":         re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "ssn":         re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b|(?i:\bssn\b\W{0,3})\d{9}\b"),
    "phone":       re.compile(
        r"(?<!\w)(?:\+?91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}\b"                          # India
        r"|(?:(?<!\w)\+?1[ .-]?)?(?:\(\d{3}\)\s?|\b\d{3}[ .-])\d{3}[ .-]\d{4}\b"  # US
    ),
    # "password: xyz" / "pwd = xyz" - the value after a credential label
    "credential": re.compile(r"(?i)\b(?:password|passwd|pwd|passcode|api[ _-]?key|secret key|token)\b\s*(?:is)?\s*[:=]?\s*(?=\S*[\d!@#$%^&*])[^\s,;]{4,}"),
    # A bare 4-6 digit number is NOT a PIN; it has to be labelled as one.
    "pin_otp_cvv": re.compile(r"(?i)\b(?:pin|otp|cvv|cvc|passcode|password)\b\s*(?:no\.?|code|number)?\s*[:#=-]?\s*\d{3,6}\b"),
}

VALIDATORS = {
    "credit_card": valid_credit_card,
    "aadhaar": valid_aadhaar,
    "ssn": valid_ssn,
    "pan": valid_pan,
    "phone": valid_phone,
}

# Words that, near an Aadhaar-shaped number, make it worth flagging even if the
# checksum fails (OCR often misreads one digit).
_AADHAAR_CONTEXT = re.compile(r"(?i)aadha+r|\buid\b|unique identification|government of india")

PII_TYPES = list(PATTERNS.keys())


@dataclass
class Finding:
    type: str
    value: str
    masked: str
    start: int
    end: int

    def to_dict(self, include_value=False):
        d = asdict(self)
        del d["start"], d["end"]  # offsets are internal (used for masking/redaction)
        if not include_value:
            d.pop("value")
        return d


def mask(value: str, keep: int = 4) -> str:
    """Replace every alphanumeric char except the last `keep` with 'X'."""
    if "@" in value:
        user, domain = value.split("@", 1)
        return user[:1] + "***@" + domain
    alnum = [i for i, ch in enumerate(value) if ch.isalnum()]
    hide = set(alnum[:-keep]) if len(alnum) > keep else set(alnum)
    return "".join("X" if i in hide else ch for i, ch in enumerate(value))


def find_pii(text: str):
    """Return a list of validated, non-overlapping Finding objects."""
    if not text:
        return []
    text = " ".join(text.split())  # normalise whitespace from OCR
    findings, taken = [], []
    for kind, pattern in PATTERNS.items():
        for m in pattern.finditer(text):
            start, end = m.span()
            value = m.group().strip()
            if any(start < e and end > s for s, e in taken):
                continue
            validator = VALIDATORS.get(kind)
            if kind == "aadhaar":
                # Printed Aadhaar is "XXXX XXXX XXXX". Flag it when (a) it has that
                # spacing and passes Verhoeff, or (b) it is labelled "Aadhaar"/"UID"
                # nearby (OCR often misreads a digit, breaking the checksum).
                labelled = bool(_AADHAAR_CONTEXT.search(text[max(0, start - 40):start]))
                spaced = not value.isdigit()
                if not (labelled or (spaced and validator(value))):
                    continue
            elif validator and not validator(value):
                continue
            if kind in ("pin_otp_cvv", "credential"):
                # keep the label ("PIN:", "password:"), hide the whole secret
                label = re.match(r"(?i)\D*?(?:pin|otp|cvv|cvc|passcode|password|passwd|pwd|key|token|code|number|no\.?)\b\s*(?:is)?\s*[:#=-]?\s*", value)
                cut = label.end() if label else 0
                masked = value[:cut] + "X" * (len(value) - cut)
            else:
                masked = mask(value)
            findings.append(Finding(kind, value, masked, start, end))
            taken.append((start, end))
    return sorted(findings, key=lambda f: f.start)


def mask_text(text: str) -> str:
    """Return text with every detected PII value replaced by its masked form."""
    text = " ".join((text or "").split())
    for f in sorted(find_pii(text), key=lambda f: f.start, reverse=True):
        text = text[:f.start] + f.masked + text[f.end:]
    return text


def pii_counts(text: str):
    """Count of findings per PII type, in PII_TYPES order (used as ML features)."""
    counts = dict.fromkeys(PII_TYPES, 0)
    for f in find_pii(text):
        counts[f.type] += 1
    return [counts[t] for t in PII_TYPES]

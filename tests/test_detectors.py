"""Unit tests for the rule-based detector.  Run:  pytest -q"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors import (find_pii, luhn_valid, mask_text, valid_ssn,  # noqa: E402
                       verhoeff_check_digit, verhoeff_valid)


def types(text):
    return [f.type for f in find_pii(text)]


def test_luhn():
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")


def test_verhoeff_roundtrip():
    base = "23456789012"
    assert verhoeff_valid(base + verhoeff_check_digit(base))
    assert not verhoeff_valid(base + str((int(verhoeff_check_digit(base)) + 1) % 10))


@pytest.mark.parametrize("text,expected", [
    ("Card 4111-1111-1111-1111", ["credit_card"]),
    ("Card 4111 1111 1111 1112", []),                     # fails Luhn
    ("SSN: 123-45-6789", ["ssn"]),
    ("SSN 000-12-3456", []),                               # area 000 never issued
    ("PAN ABCPE1234F", ["pan"]),
    ("ABCDE1234F", []),                                    # 4th char not a holder type
    ("Call +91 98765 43210", ["phone"]),
    ("Phone: +91-1234567890", []),                         # Indian mobiles start 6-9
    ("Office (480) 965-3456", ["phone"]),
    ("mail phani.s@asu.edu", ["email"]),
    ("ATM PIN: 4821", ["pin_otp_cvv"]),
    ("wifi password: sunshine42", ["credential"]),
    ("Enter your password to continue", []),            # label without a value
    ("Aadhaar: 2345 6789 0123", ["aadhaar"]),              # labelled -> kept despite checksum
    ("397788000234", []),                                  # bare digits, no label
])
def test_detection(text, expected):
    assert types(text) == expected


@pytest.mark.parametrize("text", [
    "Happy New Year 2025", "Meeting moved to room 1205", "Flight AI 302 departs 0730",
    "Invoice 2025-03-14 total 1250", "Software version 2.3.45", "Order 48213 arrives Friday",
])
def test_no_false_alarms_on_everyday_numbers(text):
    assert find_pii(text) == []


def test_ssn_rules():
    assert valid_ssn("123-45-6789")
    assert not valid_ssn("666-45-6789") and not valid_ssn("900-45-6789")
    assert not valid_ssn("123-00-6789") and not valid_ssn("123-45-0000")


def test_masking_hides_all_but_last_four():
    out = mask_text("card 4111 1111 1111 1111 and mail rahul.k@gmail.com")
    assert "4111 1111 1111 1111" not in out and out.count("1111") == 1
    assert "r***@gmail.com" in out

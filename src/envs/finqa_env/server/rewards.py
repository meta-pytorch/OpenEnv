# src/envs/finqa_env/server/rewards.py
"""
Reward computation for the FinQA environment.

Uses fuzzy numerical matching to compare predicted answers against ground truth.
Handles various formats: \boxed{}, percentages, fractions, decimals.
"""

import re
from fractions import Fraction
from typing import Optional, Tuple


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from \boxed{...} format.

    Args:
        text: Text potentially containing \boxed{answer}

    Returns:
        The extracted answer or None if not found
    """
    # Match \boxed{...} pattern
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    return None


def parse_number(text: str) -> Optional[float]:
    """
    Parse a string into a float, handling various formats.

    Handles:
    - Plain numbers: "6.118", "-3.14"
    - Percentages: "20.9%", "20.9 %"
    - Fractions: "1/2", "3/4"
    - Thousands separators: "1,234.56"
    - Negative numbers in parens: "(100)"

    Args:
        text: String to parse

    Returns:
        Float value or None if parsing fails
    """
    if text is None:
        return None

    text = text.strip()

    if not text:
        return None

    try:
        # Handle percentage
        if "%" in text:
            text = text.replace("%", "").strip()
            return float(text.replace(",", "")) / 100

        # Handle parentheses for negative numbers
        if text.startswith("(") and text.endswith(")"):
            text = "-" + text[1:-1]

        # Handle fractions (e.g., "1/2", "3/4")
        if "/" in text and not text.startswith("-"):
            try:
                return float(Fraction(text))
            except (ValueError, ZeroDivisionError):
                pass

        # Handle negative fractions
        if text.startswith("-") and "/" in text:
            try:
                return -float(Fraction(text[1:]))
            except (ValueError, ZeroDivisionError):
                pass

        # Remove thousands separators and parse
        text = text.replace(",", "")
        return float(text)

    except (ValueError, TypeError):
        return None


def normalize_answer(answer: str) -> Tuple[Optional[float], str]:
    """
    Normalize an answer string to a comparable format.

    Args:
        answer: Raw answer string

    Returns:
        Tuple of (parsed_number, cleaned_string)
    """
    if answer is None:
        return None, ""

    # Try to extract from \boxed{} first
    boxed = extract_boxed_answer(answer)
    if boxed:
        answer = boxed

    # Clean up whitespace
    answer = answer.strip()

    # Try to parse as number
    num = parse_number(answer)

    return num, answer.lower()


def compute_reward(predicted: str, ground_truth: str, tolerance: float = 0.01) -> float:
    """
    Compute reward based on answer correctness.

    Uses fuzzy numerical matching with tolerance for floating point comparison.

    Args:
        predicted: The predicted answer from the agent
        ground_truth: The expected correct answer
        tolerance: Relative tolerance for numerical comparison (default 1%)

    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    pred_num, pred_str = normalize_answer(predicted)
    truth_num, truth_str = normalize_answer(ground_truth)

    # If both are numbers, compare numerically with tolerance
    if pred_num is not None and truth_num is not None:
        # Handle zero case
        if truth_num == 0:
            return 1.0 if abs(pred_num) < 0.001 else 0.0

        # Relative tolerance check
        relative_error = abs(pred_num - truth_num) / abs(truth_num)
        if relative_error <= tolerance:
            return 1.0

        # Also check absolute tolerance for small numbers
        if abs(pred_num - truth_num) <= 0.01:
            return 1.0

        return 0.0

    # If one is a number and other isn't, not equal
    if (pred_num is None) != (truth_num is None):
        return 0.0

    # Fall back to string comparison (for non-numeric answers)
    return 1.0 if pred_str == truth_str else 0.0

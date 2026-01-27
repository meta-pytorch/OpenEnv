r"""
Test cases for FinQA reward matching with various formats and tolerances.

Tests cover:
1. LaTeX escaped percentages (\%)
2. Decimal precision matching within tolerance
3. Ratios and small numbers
4. Regular numbers and edge cases
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from envs.finqa_env.server.rewards import compute_reward, parse_number, extract_boxed_answer

# finqa labels exist in "\boxed{...}" format, e.g. "\boxed{6.280\%}", are tests are centered around this format as the labels
# Reward matching uses AND logic: both relative tolerance (1%) AND absolute difference (1.0) must pass

class TestLatexPercentages:
    """Test LaTeX escaped percentage signs in ground truth."""

    def test_latex_escaped_percentage_exact_match(self):
        """Test exact match with LaTeX escaped %."""
        assert compute_reward("6.280%", r"\boxed{6.280\%}") == 1.0
        assert compute_reward(r"6.280\%", r"\boxed{6.280\%}") == 1.0
        assert compute_reward(r"\boxed{6.280\%}", r"\boxed{6.280\%}") == 1.0

    def test_latex_escaped_percentage_within_tolerance(self):
        """Test matching within decimal tolerance."""
        assert compute_reward("6.28%", r"\boxed{6.280\%}") == 1.0
        assert compute_reward(r"\boxed{6.28%}", r"\boxed{6.280\%}") == 1.0
        assert compute_reward("0.0628", r"\boxed{6.280\%}") == 1.0

    def test_latex_percentage_with_parentheses(self):
        """Test LaTeX format with parentheses wrapper."""
        assert compute_reward("-0.606%", r"\(\boxed{-0.606\%}\)") == 1.0

    def test_latex_dollar_signs(self):
        """Test LaTeX format with dollar sign wrappers."""
        assert compute_reward("57.73", r"$\boxed{57.730}$") == 1.0


class TestDecimalPrecisionMatching:
    """Test decimal precision matching within tolerance for percentages."""

    def test_percentage_1_decimal_point_diff(self):
        """6.29% vs 6.28% should match (0.01 percentage point)."""
        assert compute_reward("6.29%", r"\boxed{6.280\%}") == 1.0
        assert compute_reward(r"\boxed{6.29\%}", r"\boxed{6.280\%}") == 1.0
        assert compute_reward("0.0629", r"\boxed{6.280\%}") == 1.0

    def test_percentage_2_decimal_points_diff(self):
        """6.30% vs 6.28% should match (0.02 percentage point)."""
        assert compute_reward("6.30%", r"\boxed{6.280\%}") == 1.0

    def test_percentage_large_diff_should_fail(self):
        """7.00% vs 6.28% should NOT match (0.72 percentage point)."""
        assert compute_reward("7.00%", r"\boxed{6.280\%}") == 0.0

    def test_percentage_1_percent_point_diff_should_fail(self):
        """7.28% vs 6.28% should NOT match (1.0 percentage point)."""
        assert compute_reward("7.28%", r"\boxed{6.280\%}") == 0.0

    def test_percentage_precision_variation(self):
        """Test different precision levels."""
        assert compute_reward("25.14%", r"\boxed{25.144\%}") == 1.0
        assert compute_reward("25.144%", r"\boxed{25.144\%}") == 1.0
        assert compute_reward("25.1%", r"\boxed{25.144\%}") == 1.0

    def test_negative_percentage_precision(self):
        """Test negative percentages within tolerance."""
        assert compute_reward("-0.61%", r"\boxed{-0.606\%}") == 1.0
        assert compute_reward("-0.606%", r"\boxed{-0.606\%}") == 1.0


class TestRatiosAndSmallNumbers:
    """Test ratio matching with appropriate decimal precision."""

    def test_ratio_exact_match(self):
        """Test exact ratio match."""
        assert compute_reward("0.232", r"\boxed{0.232}") == 1.0

    def test_ratio_1_decimal_diff(self):
        """0.233 vs 0.232 should match (0.001 diff, within tolerance)."""
        assert compute_reward("0.233", r"\boxed{0.232}") == 1.0
        assert compute_reward(r"\boxed{0.233}", r"\boxed{0.232}") == 1.0
        assert compute_reward("233/1000", r"\boxed{0.232}") == 1.0
        assert compute_reward(r"$0.233$", r"\boxed{0.232}") == 1.0

    def test_ratio_3_decimal_diff_should_fail(self):
        """0.235 vs 0.232 should NOT match (0.003 diff, exceeds relative tolerance)."""
        assert compute_reward("0.235", r"\boxed{0.232}") == 0.0

    def test_ratio_with_relative_tolerance(self):
        """Test ratios within 1% relative tolerance."""
        # 0.321 vs 0.320 = 0.31% relative error
        assert compute_reward("0.321", r"\boxed{0.320}") == 1.0
        assert compute_reward("321/1000", r"\boxed{0.320}") == 1.0
        assert compute_reward(r"\boxed{0.321}", r"\boxed{0.320}") == 1.0

    def test_small_ratios(self):
        """Test very small ratio values."""
        assert compute_reward("0.046", r"\boxed{0.046}") == 1.0
        assert compute_reward("0.0463", r"\boxed{0.046}") == 1.0


class TestRegularNumbers:
    """Test regular numbers and large values."""

    def test_negative_numbers(self):
        """Test negative number matching."""
        assert compute_reward("-77", r"\boxed{-77} million") == 1.0
        assert compute_reward(r"\boxed{-77}", r"\boxed{-77} million") == 1.0
        assert compute_reward("(77)", r"\boxed{-77} million") == 1.0

    def test_large_numbers_with_relative_tolerance(self):
        """Test large numbers must pass BOTH relative AND absolute thresholds."""
        # 1000 vs 1001 = 0.1% relative error, abs diff = 1.0, passes both
        assert compute_reward("1001", r"\boxed{1000}") == 1.0
        # 1000 vs 1009 = 0.9% relative error but abs diff = 9 > 1.0, fails
        assert compute_reward("1009", r"\boxed{1000}") == 0.0
        # 1000 vs 1011 = 1.1% relative error, should fail
        assert compute_reward("1011", r"\boxed{1000}") == 0.0

    def test_decimal_numbers(self):
        """Test decimal number matching."""
        assert compute_reward("6.118", r"\boxed{6.118}") == 1.0

    def test_thousands_separators(self):
        """Test numbers with thousand separators."""
        assert compute_reward("1,234.56", r"\boxed{1234.56}") == 1.0
        assert compute_reward("1234.56", r"\boxed{1,234.56}") == 1.0
        assert compute_reward(r"\boxed{1,234.56}", r"\boxed{1234.56}") == 1.0


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_zero_values(self):
        """Test zero value matching."""
        assert compute_reward("0", r"\boxed{0}") == 1.0
        assert compute_reward("0.0", r"\boxed{0}") == 1.0

    def test_percentage_vs_decimal(self):
        """6.28% (0.0628) should not match 6.28 (decimal)."""
        assert compute_reward("6.28%", r"\boxed{6.28}") == 0.0

    def test_fractions(self):
        """Test fraction matching."""
        assert compute_reward("1/2", r"\boxed{0.5}") == 1.0
        assert compute_reward("0.5", r"\boxed{1/2}") == 1.0
        assert compute_reward(r"\boxed{1/2}", r"\boxed{0.5}") == 1.0
        assert compute_reward(r"\boxed{0.5}", r"\boxed{1/2}") == 1.0
        assert compute_reward("50%", r"\boxed{1/2}") == 1.0

    def test_parentheses_negative(self):
        """Test negative numbers in parentheses format."""
        assert compute_reward("(100)", r"\boxed{-100}") == 1.0


class TestHelperFunctions:
    """Test helper functions used in reward computation."""

    def test_extract_boxed_answer(self):
        """Test boxed answer extraction."""
        assert extract_boxed_answer(r"\boxed{6.280\%}") == r"6.280\%"
        assert extract_boxed_answer(r"\(\boxed{-0.606\%}\)") == r"-0.606\%"
        assert extract_boxed_answer("no box here") is None

    def test_parse_number_percentages(self):
        """Test percentage parsing."""
        assert abs(parse_number("6.28%") - 0.0628) < 1e-10
        assert abs(parse_number(r"6.280\%") - 0.0628) < 1e-10
        assert abs(parse_number("-0.606%") - (-0.00606)) < 1e-10

    def test_parse_number_ratios(self):
        """Test ratio/decimal parsing."""
        assert parse_number("0.232") == 0.232
        assert parse_number("1.5") == 1.5

    def test_parse_number_fractions(self):
        """Test fraction parsing."""
        assert parse_number("1/2") == 0.5
        assert parse_number("3/4") == 0.75


class TestToleranceSettings:
    """Test the tolerance configuration."""

    def test_default_relative_tolerance(self):
        """Default relative tolerance is 1% (0.01)."""
        # 100 vs 101 = 1% relative error, should match
        assert compute_reward("101", r"\boxed{100}") == 1.0
        # 100 vs 102 = 2% relative error, should fail
        assert compute_reward("102", r"\boxed{100}") == 0.0

    def test_custom_tolerance(self):
        """Test with custom tolerance and absolute threshold parameters."""
        # With 2% tolerance but abs diff = 2 > 1.0, still fails
        assert compute_reward("102", r"\boxed{100}", tolerance=0.02) == 0.0
        # With 2% tolerance AND max_absolute_diff=3, passes
        assert compute_reward("102", r"\boxed{100}", tolerance=0.02, max_absolute_diff=3.0) == 1.0

    def test_absolute_tolerance_for_small_numbers(self):
        """Small numbers must pass both relative (1%) AND absolute (1.0) checks."""
        # 0.5 vs 0.501 = 0.001 diff, 0.2% relative, passes both
        assert compute_reward("0.501", r"\boxed{0.5}") == 1.0
        # 0.5 vs 0.506 = 0.006 diff = 1.2% relative error, fails relative
        assert compute_reward("0.506", r"\boxed{0.5}") == 0.0

    def test_absolute_tolerance_for_large_numbers(self):
        """Large numbers must pass both relative (1%) AND absolute (1.0) checks."""
        # 100 vs 100.01 = 0.01 diff, 0.01% relative, passes both
        assert compute_reward("100.01", r"\boxed{100}") == 1.0
        # 100 vs 100.5 = 0.5 diff, 0.5% relative, passes both
        assert compute_reward("100.5", r"\boxed{100}") == 1.0
        # 100 vs 102 = 2 diff, 2% relative, fails both
        assert compute_reward("102", r"\boxed{100}") == 0.0


class TestBoundaryThresholds:
    """Test boundary cases at the 2.0 threshold."""

    def test_at_threshold_exactly(self):
        """Test number exactly at 2.0 threshold."""
        assert compute_reward("2.0", r"\boxed{2.0}") == 1.0
        assert compute_reward("2.001", r"\boxed{2.0}") == 1.0

    def test_just_below_threshold(self):
        """Test number just below 2.0 threshold (uses 0.001 tolerance)."""
        # 1.999 vs 2.0 = 0.001 diff, should match with 0.001 absolute tolerance
        assert compute_reward("1.999", r"\boxed{2.0}") == 1.0

    def test_just_above_threshold(self):
        """Test number just above 2.0 threshold (uses 0.01 tolerance)."""
        # 2.001 vs 2.0 = 0.001 diff, should match with 0.01 absolute tolerance
        assert compute_reward("2.001", r"\boxed{2.0}") == 1.0


class TestScientificNotation:
    """Test scientific notation handling."""

    def test_scientific_notation_basic(self):
        """Test basic scientific notation parsing."""
        assert compute_reward("1.23e-5", r"\boxed{0.0000123}") == 1.0
        assert compute_reward("1e6", r"\boxed{1000000}") == 1.0
        assert compute_reward("0.0000123", r"\boxed{1.23e-5}") == 1.0
        assert compute_reward(r"\boxed{1e6}", r"\boxed{1000000}") == 1.0

    def test_scientific_notation_percentages(self):
        """Test scientific notation with percentages."""
        # 1.23e-3% = 0.0000123
        assert compute_reward("0.00123%", r"\boxed{1.23e-5}") == 1.0


class TestExtremeValues:
    """Test very large and very small numbers."""

    def test_very_large_numbers(self):
        """Test extremely large numbers with absolute threshold check."""
        assert compute_reward("1000000", r"\boxed{1000000}") == 1.0
        assert compute_reward("1000000", r"\boxed{1,000,000}") == 1.0
        assert compute_reward("1,000,000", r"\boxed{1000000}") == 1.0
        assert compute_reward("1e6", r"\boxed{1000000}") == 1.0
        # 1005000 vs 1000000 = 0.5% relative but abs diff = 5000 > 1.0, fails
        assert compute_reward("1005000", r"\boxed{1000000}") == 0.0
        # With custom max_absolute_diff, can pass
        assert compute_reward("1005000", r"\boxed{1000000}", max_absolute_diff=10000) == 1.0

    def test_very_small_decimals(self):
        """Test very small decimal values."""
        assert compute_reward("0.00001", r"\boxed{0.00001}") == 1.0
        # 0.000011 vs 0.00001 = 10% relative error, fails
        assert compute_reward("0.000011", r"\boxed{0.00001}") == 0.0
        # Within 1% relative: 0.00001001 vs 0.00001 = 0.1% relative, passes
        assert compute_reward("0.00001001", r"\boxed{0.00001}") == 1.0

    def test_mixed_scale_comparison(self):
        """Test comparisons across different scales."""
        # 1,000 vs 1,001 = 0.1% relative error
        assert compute_reward("1001", r"\boxed{1000}") == 1.0


class TestWhitespaceAndFormatting:
    """Test handling of whitespace and various formatting."""

    def test_extra_whitespace(self):
        """Test answers with extra whitespace."""
        assert compute_reward("  6.28%  ", r"\boxed{6.280\%}") == 1.0
        assert compute_reward("100", r"\boxed{  100  }") == 1.0

    def test_multiple_latex_wrappers(self):
        """Test various LaTeX wrapper formats."""
        # Double dollar signs
        assert compute_reward("25.14%", r"$$\boxed{25.144\%}$$") == 1.0
        # Display math mode
        assert compute_reward("0.232", r"\[\boxed{0.232}\]") == 1.0


class TestInvalidInputs:
    """Test handling of invalid or malformed inputs."""

    def test_empty_strings(self):
        """Test empty string handling."""
        assert compute_reward("", r"\boxed{100}") == 0.0
        assert compute_reward("100", "") == 0.0

    def test_non_numeric_strings(self):
        """Test non-numeric string handling."""
        assert compute_reward("abc", r"\boxed{100}") == 0.0
        assert compute_reward("100", r"\boxed{abc}") == 0.0

    def test_malformed_fractions(self):
        """Test malformed fraction handling."""
        # Division by zero should not crash
        assert parse_number("1/0") is None
        # Invalid fraction format
        assert parse_number("1/2/3") is None

    def test_mixed_formats_mismatch(self):
        """Test mismatched format types."""
        # Percentage vs plain number
        assert compute_reward("6.28", r"\boxed{6.28\%}") == 0.0
        # Fraction vs decimal (but these should match)
        assert compute_reward("0.5", r"\boxed{1/2}") == 1.0


class TestMultipleUnits:
    """Test various unit indicators."""

    def test_with_text_units(self):
        """Test numbers with text units like 'million'."""
        # The parse_number should extract just the number
        assert compute_reward("-77", r"\boxed{-77} million") == 1.0
        assert compute_reward("1.5", r"\boxed{1.5} billion") == 1.0

    def test_currency_symbols(self):
        """Test with currency symbols."""
        assert compute_reward("$100", r"\boxed{100}") == 1.0
        assert compute_reward("100", r"\boxed{$100}") == 1.0
        assert compute_reward(r"\boxed{$100}", r"\boxed{100}") == 1.0
        assert compute_reward("$100.00", r"\boxed{100}") == 1.0


class TestPrecisionEdgeCases:
    """Test edge cases in precision matching."""

    def test_leading_zeros(self):
        """Test numbers with leading zeros."""
        assert compute_reward("0.50", r"\boxed{0.5}") == 1.0
        assert compute_reward("00.5", r"\boxed{0.5}") == 1.0

    def test_percentage_boundary(self):
        """Test percentage boundary cases near 100%."""
        assert compute_reward("100%", r"\boxed{100\%}") == 1.0
        assert compute_reward("99.9%", r"\boxed{100\%}") == 1.0
        assert compute_reward("100.1%", r"\boxed{100\%}") == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

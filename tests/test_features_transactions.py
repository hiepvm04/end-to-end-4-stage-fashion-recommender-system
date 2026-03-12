"""
Unit tests cho module recsys/features/transactions.py
Kiểm tra các hàm xử lý đặc trưng giao dịch: chuyển đổi kiểu dữ liệu,
trích xuất thời gian, và tính sin/cos của tháng.
"""

import math

import numpy as np
import polars as pl
import pytest

from recsys.features.transactions import (
    calculate_month_sin_cos,
    convert_article_id_to_str,
    get_day_feature,
    get_day_of_week_feature,
    get_month_feature,
    get_year_feature,
    compute_features_transactions,
)


# ──────────────────────────────────────────────────────────────
# convert_article_id_to_str
# ──────────────────────────────────────────────────────────────

class TestConvertArticleIdToStr:
    def test_returns_string_dtype(self, sample_transactions_df):
        result = convert_article_id_to_str(sample_transactions_df)
        assert result.dtype == pl.Utf8

    def test_values_preserved_as_string(self, sample_transactions_df):
        result = convert_article_id_to_str(sample_transactions_df)
        assert "108775015" in result.to_list()

    def test_length_preserved(self, sample_transactions_df):
        result = convert_article_id_to_str(sample_transactions_df)
        assert len(result) == len(sample_transactions_df)


# ──────────────────────────────────────────────────────────────
# get_year_feature
# ──────────────────────────────────────────────────────────────

class TestGetYearFeature:
    def test_extracts_correct_year(self, sample_transactions_df):
        result = get_year_feature(sample_transactions_df)
        years = result.to_list()
        assert 2020 in years
        assert 2021 in years

    def test_length_preserved(self, sample_transactions_df):
        result = get_year_feature(sample_transactions_df)
        assert len(result) == len(sample_transactions_df)


# ──────────────────────────────────────────────────────────────
# get_month_feature
# ──────────────────────────────────────────────────────────────

class TestGetMonthFeature:
    def test_extracts_correct_month(self, sample_transactions_df):
        result = get_month_feature(sample_transactions_df)
        months = result.to_list()
        # sample có ngày 2020-09-15 và 2021-03-01
        assert 9 in months
        assert 3 in months

    def test_month_range_valid(self, sample_transactions_df):
        result = get_month_feature(sample_transactions_df)
        assert all(1 <= m <= 12 for m in result.to_list())


# ──────────────────────────────────────────────────────────────
# get_day_feature
# ──────────────────────────────────────────────────────────────

class TestGetDayFeature:
    def test_extracts_correct_day(self, sample_transactions_df):
        result = get_day_feature(sample_transactions_df)
        days = result.to_list()
        assert 15 in days
        assert 1 in days

    def test_day_range_valid(self, sample_transactions_df):
        result = get_day_feature(sample_transactions_df)
        assert all(1 <= d <= 31 for d in result.to_list())


# ──────────────────────────────────────────────────────────────
# get_day_of_week_feature
# ──────────────────────────────────────────────────────────────

class TestGetDayOfWeekFeature:
    def test_day_of_week_range_valid(self, sample_transactions_df):
        result = get_day_of_week_feature(sample_transactions_df)
        # Polars weekday: 1=Monday … 7=Sunday
        assert all(1 <= d <= 7 for d in result.to_list())

    def test_length_preserved(self, sample_transactions_df):
        result = get_day_of_week_feature(sample_transactions_df)
        assert len(result) == len(sample_transactions_df)


# ──────────────────────────────────────────────────────────────
# calculate_month_sin_cos
# ──────────────────────────────────────────────────────────────

class TestCalculateMonthSinCos:
    def test_returns_dataframe_with_two_columns(self):
        month = pl.Series([1, 6, 12])
        result = calculate_month_sin_cos(month)
        assert isinstance(result, pl.DataFrame)
        assert "month_sin" in result.columns
        assert "month_cos" in result.columns

    def test_sin_values_in_range(self):
        month = pl.Series(list(range(1, 13)))
        result = calculate_month_sin_cos(month)
        for val in result["month_sin"].to_list():
            assert -1.0 <= val <= 1.0

    def test_cos_values_in_range(self):
        month = pl.Series(list(range(1, 13)))
        result = calculate_month_sin_cos(month)
        for val in result["month_cos"].to_list():
            assert -1.0 <= val <= 1.0

    def test_known_value_month_3(self):
        """Tháng 3: sin(3 * 2π/12) = sin(π/2) ≈ 1.0"""
        month = pl.Series([3])
        result = calculate_month_sin_cos(month)
        assert math.isclose(result["month_sin"][0], 1.0, abs_tol=1e-9)

    def test_known_value_month_12(self):
        """Tháng 12: sin(12 * 2π/12) = sin(2π) ≈ 0.0"""
        month = pl.Series([12])
        result = calculate_month_sin_cos(month)
        assert math.isclose(result["month_sin"][0], 0.0, abs_tol=1e-9)

    def test_cos_month_1(self):
        """Tháng 1: cos(1 * 2π/12) = cos(π/6) ≈ √3/2"""
        month = pl.Series([1])
        result = calculate_month_sin_cos(month)
        expected = math.cos(2 * math.pi / 12)
        assert math.isclose(result["month_cos"][0], expected, abs_tol=1e-9)


# ──────────────────────────────────────────────────────────────
# compute_features_transactions (integration-style)
# ──────────────────────────────────────────────────────────────

class TestComputeFeaturesTransactions:
    def test_output_has_article_id_as_string(self, sample_transactions_df):
        result = compute_features_transactions(sample_transactions_df)
        assert result["article_id"].dtype == pl.Utf8

    def test_output_has_year_column(self, sample_transactions_df):
        result = compute_features_transactions(sample_transactions_df)
        assert "year" in result.columns

    def test_output_has_month_column(self, sample_transactions_df):
        result = compute_features_transactions(sample_transactions_df)
        assert "month" in result.columns

    def test_output_has_day_column(self, sample_transactions_df):
        result = compute_features_transactions(sample_transactions_df)
        assert "day" in result.columns

    def test_output_has_day_of_week_column(self, sample_transactions_df):
        result = compute_features_transactions(sample_transactions_df)
        assert "day_of_week" in result.columns

    def test_t_dat_converted_to_epoch_ms(self, sample_transactions_df):
        result = compute_features_transactions(sample_transactions_df)
        # Sau khi convert, t_dat phải là Int64
        assert result["t_dat"].dtype == pl.Int64

    def test_row_count_preserved(self, sample_transactions_df):
        result = compute_features_transactions(sample_transactions_df)
        assert len(result) == len(sample_transactions_df)

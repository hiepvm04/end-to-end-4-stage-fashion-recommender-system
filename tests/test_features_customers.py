"""
Unit tests cho module recsys/features/customers.py
Kiểm tra các hàm xử lý đặc trưng khách hàng: điền null, nhóm tuổi, v.v.
"""

import polars as pl
import pytest

from recsys.features.customers import (
    DatasetSampler,
    compute_features_customers,
    create_age_group,
    drop_na_age,
    fill_missing_club_member_status,
)
from recsys.config import CustomerDatasetSize


# ──────────────────────────────────────────────────────────────
# fill_missing_club_member_status
# ──────────────────────────────────────────────────────────────

class TestFillMissingClubMemberStatus:
    def test_null_values_replaced_with_absent(self, sample_customers_df):
        result = fill_missing_club_member_status(sample_customers_df)
        assert result["club_member_status"].null_count() == 0

    def test_null_replaced_by_absent_string(self, sample_customers_df):
        result = fill_missing_club_member_status(sample_customers_df)
        assert "ABSENT" in result["club_member_status"].to_list()

    def test_existing_values_unchanged(self, sample_customers_df):
        result = fill_missing_club_member_status(sample_customers_df)
        assert "ACTIVE" in result["club_member_status"].to_list()
        assert "INACTIVE" in result["club_member_status"].to_list()


# ──────────────────────────────────────────────────────────────
# drop_na_age
# ──────────────────────────────────────────────────────────────

class TestDropNaAge:
    def test_rows_with_null_age_are_removed(self, sample_customers_df):
        # sample_customers_df có 1 hàng age=null (C003)
        result = drop_na_age(sample_customers_df)
        assert result["age"].null_count() == 0

    def test_row_count_decreases(self, sample_customers_df):
        original_count = len(sample_customers_df)
        result = drop_na_age(sample_customers_df)
        assert len(result) < original_count

    def test_non_null_rows_kept(self, sample_customers_df):
        result = drop_na_age(sample_customers_df)
        assert "C001" in result["customer_id"].to_list()
        assert "C002" in result["customer_id"].to_list()

    def test_null_row_removed(self, sample_customers_df):
        result = drop_na_age(sample_customers_df)
        # C003 có age=null phải bị loại
        assert "C003" not in result["customer_id"].to_list()


# ──────────────────────────────────────────────────────────────
# create_age_group
# ──────────────────────────────────────────────────────────────

class TestCreateAgeGroup:
    def _apply_age_group(self, ages: list) -> list:
        df = pl.DataFrame({"age": ages})
        return df.with_columns(create_age_group())["age_group"].to_list()

    def test_child_age_group(self):
        groups = self._apply_age_group([10])
        assert groups[0] == "0-18"

    def test_young_adult_age_group(self):
        groups = self._apply_age_group([22])
        assert groups[0] == "19-25"

    def test_adult_age_group(self):
        groups = self._apply_age_group([30])
        assert groups[0] == "26-35"

    def test_middle_age_group(self):
        groups = self._apply_age_group([40])
        assert groups[0] == "36-45"

    def test_senior_age_group(self):
        groups = self._apply_age_group([50])
        assert groups[0] == "46-55"

    def test_older_age_group(self):
        groups = self._apply_age_group([60])
        assert groups[0] == "56-65"

    def test_elderly_age_group(self):
        groups = self._apply_age_group([70])
        assert groups[0] == "66+"

    def test_boundary_age_18(self):
        groups = self._apply_age_group([18])
        assert groups[0] == "0-18"

    def test_boundary_age_19(self):
        groups = self._apply_age_group([19])
        assert groups[0] == "19-25"


# ──────────────────────────────────────────────────────────────
# compute_features_customers
# ──────────────────────────────────────────────────────────────

class TestComputeFeaturesCustomers:
    def test_output_has_correct_columns(self, sample_customers_df):
        result = compute_features_customers(sample_customers_df)
        expected_columns = {
            "customer_id", "club_member_status", "age", "postal_code", "age_group"
        }
        assert set(result.columns) == expected_columns

    def test_age_column_is_float64(self, sample_customers_df):
        result = compute_features_customers(sample_customers_df)
        assert result["age"].dtype == pl.Float64

    def test_no_null_age_in_output(self, sample_customers_df):
        result = compute_features_customers(sample_customers_df)
        assert result["age"].null_count() == 0

    def test_no_null_club_member_status_in_output(self, sample_customers_df):
        result = compute_features_customers(sample_customers_df)
        assert result["club_member_status"].null_count() == 0

    def test_missing_column_raises_value_error(self):
        df_missing = pl.DataFrame(
            {"customer_id": ["C001"], "club_member_status": ["ACTIVE"]}
        )
        with pytest.raises(ValueError, match="age"):
            compute_features_customers(df_missing)

    def test_missing_multiple_columns_raises_value_error(self):
        df_empty = pl.DataFrame({"some_column": ["x"]})
        with pytest.raises(ValueError):
            compute_features_customers(df_empty)

    def test_age_group_column_populated(self, sample_customers_df_no_nulls):
        result = compute_features_customers(sample_customers_df_no_nulls)
        assert result["age_group"].null_count() == 0
        assert all(g is not None for g in result["age_group"].to_list())


# ──────────────────────────────────────────────────────────────
# DatasetSampler
# ──────────────────────────────────────────────────────────────

class TestDatasetSampler:
    def test_get_supported_sizes_returns_dict(self):
        sizes = DatasetSampler.get_supported_sizes()
        assert isinstance(sizes, dict)

    def test_supported_sizes_has_all_enum_values(self):
        sizes = DatasetSampler.get_supported_sizes()
        assert CustomerDatasetSize.SMALL in sizes
        assert CustomerDatasetSize.MEDIUM in sizes
        assert CustomerDatasetSize.LARGE in sizes

    def test_small_size_is_1000(self):
        sizes = DatasetSampler.get_supported_sizes()
        assert sizes[CustomerDatasetSize.SMALL] == 1_000

    def test_medium_size_is_5000(self):
        sizes = DatasetSampler.get_supported_sizes()
        assert sizes[CustomerDatasetSize.MEDIUM] == 5_000

    def test_large_size_is_50000(self):
        sizes = DatasetSampler.get_supported_sizes()
        assert sizes[CustomerDatasetSize.LARGE] == 50_000

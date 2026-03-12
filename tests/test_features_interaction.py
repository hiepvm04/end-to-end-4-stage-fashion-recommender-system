"""
Unit tests cho module recsys/features/interaction.py
Kiểm tra hàm generate_interaction_data: schema output, giá trị score hợp lệ,
và xử lý trường hợp đặc biệt (empty input, prev_article_id).
"""

import polars as pl
import pytest

from recsys.features.interaction import generate_interaction_data


# Cột chuẩn theo schema của Feature Group
EXPECTED_COLUMNS = {"t_dat", "customer_id", "article_id", "interaction_score", "prev_article_id"}
VALID_INTERACTION_SCORES = {0, 1, 2}


# ──────────────────────────────────────────────────────────────
# Test với DataFrame rỗng
# ──────────────────────────────────────────────────────────────

class TestGenerateInteractionDataEmptyInput:
    def test_empty_input_returns_dataframe(self):
        empty_df = pl.DataFrame(
            schema={
                "t_dat": pl.Int64,
                "customer_id": pl.Utf8,
                "article_id": pl.Utf8,
                "price": pl.Float64,
                "sales_channel_id": pl.Int64,
            }
        )
        result = generate_interaction_data(empty_df)
        assert isinstance(result, pl.DataFrame)

    def test_empty_input_returns_empty_dataframe(self):
        empty_df = pl.DataFrame(
            schema={
                "t_dat": pl.Int64,
                "customer_id": pl.Utf8,
                "article_id": pl.Utf8,
                "price": pl.Float64,
                "sales_channel_id": pl.Int64,
            }
        )
        result = generate_interaction_data(empty_df)
        assert len(result) == 0


# ──────────────────────────────────────────────────────────────
# Test với dữ liệu giả lập
# ──────────────────────────────────────────────────────────────

class TestGenerateInteractionDataWithData:
    def test_output_has_required_columns(self, sample_interaction_transactions_df):
        result = generate_interaction_data(sample_interaction_transactions_df)
        assert EXPECTED_COLUMNS.issubset(set(result.columns))

    def test_interaction_scores_are_valid(self, sample_interaction_transactions_df):
        result = generate_interaction_data(sample_interaction_transactions_df)
        if len(result) > 0:
            scores = set(result["interaction_score"].to_list())
            assert scores.issubset(VALID_INTERACTION_SCORES)

    def test_output_is_non_empty(self, sample_interaction_transactions_df):
        result = generate_interaction_data(sample_interaction_transactions_df)
        # Phải có ít nhất 1 interaction được tạo ra (click / purchase / ignore)
        assert len(result) > 0

    def test_all_customer_ids_present(self, sample_interaction_transactions_df):
        result = generate_interaction_data(sample_interaction_transactions_df)
        if len(result) > 0:
            customer_ids_in_result = set(result["customer_id"].to_list())
            assert "C001" in customer_ids_in_result

    def test_purchases_have_score_2(self, sample_interaction_transactions_df):
        result = generate_interaction_data(sample_interaction_transactions_df)
        if len(result) > 0:
            purchase_rows = result.filter(pl.col("interaction_score") == 2)
            # Phải có ít nhất 1 purchase (vì input có 2 transactions)
            assert len(purchase_rows) >= 1

    def test_first_row_per_customer_has_start_prev_article(self, sample_interaction_transactions_df):
        """
        Hàng đầu tiên (theo t_dat) của mỗi customer phải có prev_article_id = 'START'.
        """
        result = generate_interaction_data(sample_interaction_transactions_df)
        if len(result) > 0:
            # Lấy hàng đầu tiên của customer C001
            c001_rows = result.filter(pl.col("customer_id") == "C001").sort("t_dat")
            first_row = c001_rows.row(0, named=True)
            assert first_row["prev_article_id"] == "START"

    def test_output_sorted_by_customer_and_time(self, sample_interaction_transactions_df):
        """
        Output phải được sắp xếp theo customer_id rồi t_dat.
        """
        result = generate_interaction_data(sample_interaction_transactions_df)
        if len(result) > 1:
            # Kiểm tra t_dat tăng dần trong cùng customer
            for customer_id in result["customer_id"].unique().to_list():
                customer_df = result.filter(pl.col("customer_id") == customer_id)
                timestamps = customer_df["t_dat"].to_list()
                assert timestamps == sorted(timestamps)

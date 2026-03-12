"""
Shared pytest fixtures dùng xuyên suốt các test files.
Tất cả fixtures đều dùng dữ liệu giả lập – không gọi Hopsworks hay OpenAI.
"""

from datetime import date

import polars as pl
import pytest


@pytest.fixture
def sample_customers_df() -> pl.DataFrame:
    """
    DataFrame giả lập dữ liệu khách hàng với đầy đủ các trường cần thiết.
    - Bao gồm 1 hàng có age = null để test `drop_na_age`
    - Bao gồm 1 hàng có club_member_status = null để test `fill_missing_club_member_status`
    """
    return pl.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003", "C004", "C005"],
            "club_member_status": ["ACTIVE", None, "INACTIVE", "ACTIVE", "PRE-CREATE"],
            "age": [22.0, 35.0, None, 55.0, 17.0],
            "postal_code": ["10000", "20000", "30000", "40000", "50000"],
        }
    )


@pytest.fixture
def sample_customers_df_no_nulls() -> pl.DataFrame:
    """DataFrame khách hàng không có giá trị null."""
    return pl.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003"],
            "club_member_status": ["ACTIVE", "INACTIVE", "ACTIVE"],
            "age": [22.0, 35.0, 55.0],
            "postal_code": ["10000", "20000", "30000"],
        }
    )


@pytest.fixture
def sample_articles_df() -> pl.DataFrame:
    """
    DataFrame giả lập dữ liệu sản phẩm thời trang H&M.
    Bao gồm đủ các cột cần thiết để tạo article_description.
    """
    return pl.DataFrame(
        {
            "article_id": [108775015, 108775044, 110065001],
            "product_code": [108775, 108775, 110065],
            "prod_name": ["Strap top", "Strap top", "OP T-shirt (Pullov)"],
            "product_type_name": ["Vest top", "Vest top", "Sweater"],
            "product_group_name": ["Garment Upper body", "Garment Upper body", "Garment Upper body"],
            "graphical_appearance_name": ["Solid", "Stripe", "Solid"],
            "colour_group_name": ["Black", "White", "Grey"],
            "perceived_colour_value_name": ["Dark", "Light", "Medium"],
            "perceived_colour_master_name": ["Black", "White", "Grey"],
            "department_name": ["Jersey Basic", "Jersey Basic", "Knitwear"],
            "index_name": ["Ladieswear", "Ladieswear", "Divided"],
            "index_group_name": ["Ladieswear", "Ladieswear", "Divided"],
            "section_name": ["Womens Everyday Basics", "Womens Casual", "Young Casual"],
            "garment_group_name": ["Jersey Fancy", "Jersey Fancy", "Knitwear"],
            "detail_desc": ["A vest top", None, "A comfortable sweater"],
        }
    )


@pytest.fixture
def sample_transactions_df() -> pl.DataFrame:
    """
    DataFrame giả lập giao dịch với cột t_dat là kiểu Date (Polars).
    """
    return pl.DataFrame(
        {
            "t_dat": pl.Series(
                [date(2020, 9, 15), date(2020, 9, 16), date(2021, 3, 1)]
            ).cast(pl.Datetime),
            "customer_id": ["C001", "C002", "C001"],
            "article_id": pl.Series([108775015, 108775044, 110065001]),
            "price": [0.025, 0.015, 0.030],
            "sales_channel_id": [2, 1, 2],
        }
    )


@pytest.fixture
def sample_interaction_transactions_df() -> pl.DataFrame:
    """
    DataFrame giao dịch nhỏ dùng để test hàm generate_interaction_data.
    t_dat đã ở dạng Int64 (epoch milliseconds) – giống format sau compute_features_transactions.
    """
    return pl.DataFrame(
        {
            "t_dat": pl.Series([1_600_000_000_000, 1_600_100_000_000]),
            "customer_id": ["C001", "C001"],
            "article_id": ["108775015", "108775044"],
            "price": [0.025, 0.015],
            "sales_channel_id": [2, 1],
        }
    )

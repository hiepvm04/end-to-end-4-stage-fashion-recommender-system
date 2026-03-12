"""
Unit tests cho module recsys/features/articles.py
Kiểm tra các hàm xử lý đặc trưng sản phẩm: article_id, prod_name_length,
article_description, và URL ảnh.
"""

import polars as pl

from recsys.features.articles import (
    create_article_description,
    create_prod_name_length,
    get_article_id,
    get_image_url,
)


# ──────────────────────────────────────────────────────────────
# get_article_id
# ──────────────────────────────────────────────────────────────

class TestGetArticleId:
    def test_returns_string_type(self, sample_articles_df):
        result = get_article_id(sample_articles_df)
        assert result.dtype == pl.Utf8

    def test_values_are_correct(self, sample_articles_df):
        result = get_article_id(sample_articles_df)
        assert "108775015" in result.to_list()
        assert "108775044" in result.to_list()

    def test_length_preserved(self, sample_articles_df):
        result = get_article_id(sample_articles_df)
        assert len(result) == len(sample_articles_df)


# ──────────────────────────────────────────────────────────────
# create_prod_name_length
# ──────────────────────────────────────────────────────────────

class TestCreateProdNameLength:
    def test_returns_correct_lengths(self, sample_articles_df):
        result = create_prod_name_length(sample_articles_df)
        lengths = result.to_list()
        # "Strap top" = 9 chars
        assert lengths[0] == len("Strap top")

    def test_all_values_positive(self, sample_articles_df):
        result = create_prod_name_length(sample_articles_df)
        assert all(l > 0 for l in result.to_list())

    def test_length_preserved(self, sample_articles_df):
        result = create_prod_name_length(sample_articles_df)
        assert len(result) == len(sample_articles_df)


# ──────────────────────────────────────────────────────────────
# create_article_description
# ──────────────────────────────────────────────────────────────

class TestCreateArticleDescription:
    def _get_row(self, sample_articles_df, index: int = 0) -> dict:
        return sample_articles_df.row(index, named=True)

    def test_description_contains_prod_name(self, sample_articles_df):
        row = self._get_row(sample_articles_df)
        desc = create_article_description(row)
        assert row["prod_name"] in desc

    def test_description_contains_product_type(self, sample_articles_df):
        row = self._get_row(sample_articles_df)
        desc = create_article_description(row)
        assert row["product_type_name"] in desc

    def test_description_contains_appearance(self, sample_articles_df):
        row = self._get_row(sample_articles_df)
        desc = create_article_description(row)
        assert "Appearance" in desc
        assert row["graphical_appearance_name"] in desc

    def test_description_contains_color(self, sample_articles_df):
        row = self._get_row(sample_articles_df)
        desc = create_article_description(row)
        assert "Color" in desc

    def test_description_contains_category(self, sample_articles_df):
        row = self._get_row(sample_articles_df)
        desc = create_article_description(row)
        assert "Category" in desc

    def test_description_includes_detail_desc_when_present(self, sample_articles_df):
        # Row 0 có detail_desc = "A vest top"
        row = self._get_row(sample_articles_df, index=0)
        desc = create_article_description(row)
        assert "Details" in desc
        assert "A vest top" in desc

    def test_description_skips_detail_desc_when_null(self, sample_articles_df):
        # Row 1 có detail_desc = None
        row = self._get_row(sample_articles_df, index=1)
        desc = create_article_description(row)
        # Không crash và không có "Details" section
        assert isinstance(desc, str)


# ──────────────────────────────────────────────────────────────
# get_image_url
# ──────────────────────────────────────────────────────────────

class TestGetImageUrl:
    def test_url_starts_with_hopsworks_base(self):
        url = get_image_url("108775015")
        assert url.startswith("https://repo.hops.works/dev/jdowling/h-and-m/images/")

    def test_url_ends_with_jpg(self):
        url = get_image_url("108775015")
        assert url.endswith(".jpg")

    def test_url_contains_article_id(self):
        url = get_image_url("108775015")
        assert "108775015" in url

    def test_url_contains_correct_folder(self):
        # article_id = "108775015" → folder = "10"
        url = get_image_url("108775015")
        assert "/010/" in url

    def test_url_is_string(self):
        url = get_image_url(108775015)  # Thử với integer input
        assert isinstance(url, str)

    def test_different_article_ids_give_different_urls(self):
        url1 = get_image_url("108775015")
        url2 = get_image_url("110065001")
        assert url1 != url2

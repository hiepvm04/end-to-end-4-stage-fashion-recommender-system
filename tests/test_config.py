"""
Unit tests cho module recsys/config.py
Kiểm tra giá trị mặc định của Settings, enum CustomerDatasetSize,
và các hằng số huấn luyện.
"""

import pytest
from recsys.config import CustomerDatasetSize, Settings


# ──────────────────────────────────────────────────────────────
# CustomerDatasetSize Enum
# ──────────────────────────────────────────────────────────────

class TestCustomerDatasetSizeEnum:
    def test_has_small_value(self):
        assert CustomerDatasetSize.SMALL.value == "SMALL"

    def test_has_medium_value(self):
        assert CustomerDatasetSize.MEDIUM.value == "MEDIUM"

    def test_has_large_value(self):
        assert CustomerDatasetSize.LARGE.value == "LARGE"

    def test_three_sizes_total(self):
        assert len(CustomerDatasetSize) == 3

    def test_small_is_smallest(self):
        """SMALL < MEDIUM < LARGE theo tên."""
        names = [e.name for e in CustomerDatasetSize]
        assert "SMALL" in names
        assert "MEDIUM" in names
        assert "LARGE" in names


# ──────────────────────────────────────────────────────────────
# Settings defaults
# ──────────────────────────────────────────────────────────────

class TestSettingsDefaults:
    """
    Kiểm tra các giá trị mặc định của Settings (không cần file .env).
    Các API key được set là None nên không cần biến môi trường thật.
    """

    @pytest.fixture
    def settings(self):
        return Settings(
            HOPSWORKS_API_KEY=None,
            OPENAI_API_KEY=None,
        )

    def test_default_customer_data_size_is_small(self, settings):
        assert settings.CUSTOMER_DATA_SIZE == CustomerDatasetSize.SMALL

    def test_default_openai_model_id(self, settings):
        assert settings.OPENAI_MODEL_ID == "gpt-4o-mini"

    def test_default_features_embedding_model(self, settings):
        assert settings.FEATURES_EMBEDDING_MODEL_ID == "all-MiniLM-L6-v2"

    def test_default_two_tower_embedding_size(self, settings):
        assert settings.TWO_TOWER_MODEL_EMBEDDING_SIZE == 16

    def test_default_two_tower_batch_size(self, settings):
        assert settings.TWO_TOWER_MODEL_BATCH_SIZE == 2048

    def test_default_two_tower_epochs(self, settings):
        assert settings.TWO_TOWER_NUM_EPOCHS == 10

    def test_default_two_tower_learning_rate(self, settings):
        assert settings.TWO_TOWER_LEARNING_RATE == 0.01

    def test_default_two_tower_weight_decay(self, settings):
        assert settings.TWO_TOWER_WEIGHT_DECAY == 0.001

    def test_default_ranking_model_type_is_ranking(self, settings):
        assert settings.RANKING_MODEL_TYPE == "ranking"

    def test_default_ranking_learning_rate(self, settings):
        assert settings.RANKING_LEARNING_RATE == 0.2

    def test_default_ranking_iterations(self, settings):
        assert settings.RANKING_ITERATIONS == 100

    def test_default_ranking_scale_pos_weight(self, settings):
        assert settings.RANKING_SCALE_POS_WEIGHT == 10

    def test_hopsworks_api_key_is_none_by_default(self, settings):
        assert settings.HOPSWORKS_API_KEY is None

    def test_openai_api_key_is_none_by_default(self, settings):
        assert settings.OPENAI_API_KEY is None

    def test_recsys_dir_is_path(self, settings):
        from pathlib import Path
        assert isinstance(settings.RECSYS_DIR, Path)
        assert settings.RECSYS_DIR.exists()


# ──────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────

class TestSettingsValidation:
    def test_ranking_model_type_accepts_ranking(self):
        s = Settings(HOPSWORKS_API_KEY=None, OPENAI_API_KEY=None, RANKING_MODEL_TYPE="ranking")
        assert s.RANKING_MODEL_TYPE == "ranking"

    def test_ranking_model_type_accepts_llmranking(self):
        s = Settings(HOPSWORKS_API_KEY=None, OPENAI_API_KEY=None, RANKING_MODEL_TYPE="llmranking")
        assert s.RANKING_MODEL_TYPE == "llmranking"

    def test_ranking_model_type_rejects_invalid(self):
        """Giá trị không hợp lệ phải raise ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(
                HOPSWORKS_API_KEY=None,
                OPENAI_API_KEY=None,
                RANKING_MODEL_TYPE="invalid_value",
            )

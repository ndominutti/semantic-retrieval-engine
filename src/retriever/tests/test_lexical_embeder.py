import os
import tempfile
from unittest.mock import patch

import joblib
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.retriever.src.batch_embedings.exceptions import (
    EmptyDataFrameError,
    MissingColumnsError,
)
from src.retriever.src.batch_embedings.generators.lexical_embeder import (
    TDIDFLexicalEmbeder,
)


class TestTDIDFLexicalEmbeder:
    """Test suite for TDIDFLexicalEmbeder class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing file operations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir

    @pytest.fixture
    def embeder(self, temp_dir):
        """Create a TDIDFLexicalEmbeder instance for testing."""
        return TDIDFLexicalEmbeder(save_dir=temp_dir)

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "title": ["Product A", "Product B", "Product C"],
                "description": ["Great product", "Amazing item", "Wonderful tool"],
                "category": ["Electronics", "Books", "Tools"],
                "price": [10.99, 15.50, 8.75],
            }
        )

    def test_init(self, temp_dir):
        """Test initialization of TDIDFLexicalEmbeder."""
        embeder = TDIDFLexicalEmbeder(save_dir=temp_dir)

        assert isinstance(embeder.vectorizer, TfidfVectorizer)
        assert embeder.save_dir == temp_dir

    def test_fit_transform_basic(self, embeder, sample_dataframe):
        """Test basic fit_transform functionality."""
        cols_to_embed = ["title", "description"]
        result = embeder.fit_transform(sample_dataframe, cols_to_embed)

        # Check return type
        assert isinstance(result, csr_matrix)

        # Check dimensions - should have same number of rows as DataFrame
        assert result.shape[0] == len(sample_dataframe)

        # Check that vectorizer was fitted
        assert hasattr(embeder.vectorizer, "vocabulary_")
        assert len(embeder.vectorizer.vocabulary_) > 0

    def test_fit_transform_single_column(self, embeder, sample_dataframe):
        """Test fit_transform with a single column."""
        cols_to_embed = ["title"]
        result = embeder.fit_transform(sample_dataframe, cols_to_embed)

        assert isinstance(result, csr_matrix)
        assert result.shape[0] == len(sample_dataframe)

    def test_fit_transform_all_columns(self, embeder, sample_dataframe):
        """Test fit_transform with all text columns."""
        cols_to_embed = ["title", "description", "category"]
        result = embeder.fit_transform(sample_dataframe, cols_to_embed)

        assert isinstance(result, csr_matrix)
        assert result.shape[0] == len(sample_dataframe)

    def test_fit_transform_missing_columns_error(self, embeder, sample_dataframe):
        """Test that MissingColumnsError is raised for non-existent columns."""
        cols_to_embed = ["title", "nonexistent_column"]

        with pytest.raises(MissingColumnsError) as exc_info:
            embeder.fit_transform(sample_dataframe, cols_to_embed)

        assert "Missing columns in DataFrame: ['nonexistent_column']" in str(
            exc_info.value
        )

    def test_fit_transform_multiple_missing_columns(self, embeder, sample_dataframe):
        """Test error message with multiple missing columns."""
        cols_to_embed = ["title", "missing1", "missing2"]

        with pytest.raises(MissingColumnsError) as exc_info:
            embeder.fit_transform(sample_dataframe, cols_to_embed)

        error_msg = str(exc_info.value)
        assert "missing1" in error_msg
        assert "missing2" in error_msg

    def test_fit_transform_with_nan_values(self, embeder):
        """Test fit_transform handles NaN values correctly."""
        df_with_nan = pd.DataFrame(
            {
                "title": ["Product A", None, "Product C"],
                "description": ["Great product", "Amazing item", None],
            }
        )

        cols_to_embed = ["title", "description"]
        result = embeder.fit_transform(df_with_nan, cols_to_embed)

        assert isinstance(result, csr_matrix)
        assert result.shape[0] == len(df_with_nan)

    def test_fit_transform_empty_dataframe_error(self, embeder):
        """Test that EmptyDataFrameError is raised for empty DataFrame."""
        empty_df = pd.DataFrame({"title": [], "description": []})
        cols_to_embed = ["title", "description"]

        with pytest.raises(EmptyDataFrameError) as exc_info:
            embeder.fit_transform(empty_df, cols_to_embed)

        assert (
            "DataFrame is empty" in str(exc_info.value)
            or "empty" in str(exc_info.value).lower()
        )

    def test_empty_dataframe_with_valid_columns(self, embeder):
        """Test EmptyDataFrameError when DataFrame is empty but columns exist."""
        empty_df = pd.DataFrame(columns=["title", "description"])
        cols_to_embed = ["title", "description"]

        with pytest.raises(EmptyDataFrameError):
            embeder.fit_transform(empty_df, cols_to_embed)

    def test_fit_transform_empty_dataframe_different_columns(self, embeder):
        """Test EmptyDataFrameError with different column combinations."""
        empty_df = pd.DataFrame({"col1": [], "col2": [], "col3": []})

        # Test with single column
        with pytest.raises(EmptyDataFrameError):
            embeder.fit_transform(empty_df, ["col1"])

        # Test with multiple columns
        with pytest.raises(EmptyDataFrameError):
            embeder.fit_transform(empty_df, ["col1", "col2"])

    def test_save_functionality(self, embeder, sample_dataframe, temp_dir):
        """Test save method creates files correctly."""
        cols_to_embed = ["title", "description"]
        vectors = embeder.fit_transform(sample_dataframe, cols_to_embed)

        embeder.save(vectors)

        # Check that files were created
        vectorizer_path = os.path.join(temp_dir, "tfidf_vectorizer.joblib")
        matrix_path = os.path.join(temp_dir, "tfidf_matrix.joblib")

        assert os.path.exists(vectorizer_path)
        assert os.path.exists(matrix_path)

    def test_save_and_load_consistency(self, embeder, sample_dataframe, temp_dir):
        """Test that saved objects can be loaded and are consistent."""
        cols_to_embed = ["title", "description"]
        original_vectors = embeder.fit_transform(sample_dataframe, cols_to_embed)

        embeder.save(original_vectors)

        # Load saved objects
        vectorizer_path = os.path.join(temp_dir, "tfidf_vectorizer.joblib")
        matrix_path = os.path.join(temp_dir, "tfidf_matrix.joblib")

        loaded_vectorizer = joblib.load(vectorizer_path)
        loaded_vectors = joblib.load(matrix_path)

        # Check consistency
        assert type(loaded_vectorizer) is type(embeder.vectorizer)
        assert loaded_vectorizer.vocabulary_ == embeder.vectorizer.vocabulary_
        assert (loaded_vectors != original_vectors).nnz == 0  # Compare sparse matrices

    @patch("joblib.dump")
    def test_save_calls_joblib_dump(self, mock_dump, embeder, sample_dataframe):
        """Test that save method calls joblib.dump with correct parameters."""
        cols_to_embed = ["title", "description"]
        vectors = embeder.fit_transform(sample_dataframe, cols_to_embed)

        embeder.save(vectors)

        # Check that joblib.dump was called twice
        assert mock_dump.call_count == 2

        # Check call arguments
        calls = mock_dump.call_args_list

        # First call should save vectorizer
        assert calls[0][0][0] == embeder.vectorizer
        assert calls[0][0][1].endswith("tfidf_vectorizer.joblib")

        # Second call should save vectors
        assert calls[1][0][0] is vectors
        assert calls[1][0][1].endswith("tfidf_matrix.joblib")

    def test_text_combination_correctness(self, embeder):
        """Test that text from multiple columns is combined correctly."""
        df = pd.DataFrame({"col1": ["hello", "world"], "col2": ["foo", "bar"]})

        cols_to_embed = ["col1", "col2"]

        # Mock the vectorizer to capture what text is passed to it
        with patch.object(embeder.vectorizer, "fit_transform") as mock_fit_transform:
            mock_fit_transform.return_value = csr_matrix((2, 10))  # Mock return

            embeder.fit_transform(df, cols_to_embed)

            # Check that combined text was passed correctly
            called_with = mock_fit_transform.call_args[0][0]
            expected = ["hello foo", "world bar"]

            assert list(called_with) == expected

    def test_different_save_directories(self):
        """Test that different save directories work correctly."""
        dir1 = tempfile.mkdtemp()
        dir2 = tempfile.mkdtemp()

        try:
            embeder1 = TDIDFLexicalEmbeder(save_dir=dir1)
            embeder2 = TDIDFLexicalEmbeder(save_dir=dir2)

            assert embeder1.save_dir == dir1
            assert embeder2.save_dir == dir2
            assert embeder1.save_dir != embeder2.save_dir
        finally:
            # Clean up
            import shutil

            shutil.rmtree(dir1, ignore_errors=True)
            shutil.rmtree(dir2, ignore_errors=True)


# Integration tests
class TestTDIDFLexicalEmbederIntegration:
    """Integration tests for TDIDFLexicalEmbeder."""

    def test_full_workflow(self, tmp_path):
        """Test the complete workflow: init -> fit_transform -> save."""
        embeder = TDIDFLexicalEmbeder(save_dir=str(tmp_path))

        # Create test data
        df = pd.DataFrame(
            {
                "product_name": ["iPhone 13", "Samsung Galaxy", "Google Pixel"],
                "description": [
                    "Latest Apple phone",
                    "Android smartphone",
                    "Google phone",
                ],
                "brand": ["Apple", "Samsung", "Google"],
            }
        )

        # Fit and transform
        cols_to_embed = ["product_name", "description"]
        vectors = embeder.fit_transform(df, cols_to_embed)

        # Save
        embeder.save(vectors)

        # Verify results
        assert vectors.shape[0] == len(df)
        assert len(embeder.vectorizer.vocabulary_) > 0

        # Verify files exist
        assert (tmp_path / "tfidf_vectorizer.joblib").exists()
        assert (tmp_path / "tfidf_matrix.joblib").exists()


# Parametrized tests for edge cases
class TestTDIDFLexicalEmbederEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.parametrize(
        "cols_to_embed",
        [["title"], ["title", "description"], ["title", "description", "category"]],
    )
    def test_different_column_combinations(self, cols_to_embed, tmp_path):
        """Test with different combinations of columns."""
        embeder = TDIDFLexicalEmbeder(save_dir=str(tmp_path))

        df = pd.DataFrame(
            {
                "title": ["Al", "nnB", "Ckk"],
                "description": ["X", "Y", "Z"],
                "category": ["10", "2", "3"],
            }
        )

        result = embeder.fit_transform(df, cols_to_embed)
        assert result.shape[0] == len(df)

    @pytest.mark.parametrize(
        "text_data",
        [
            ["short", "text", "here"],
            ["this is a much longer piece of text with many words"] * 3,
            ["special!@#$%^&*()characters", "números123", "mixed_text-here"],
        ],
    )
    def test_different_text_content(self, text_data, tmp_path):
        """Test with different types of text content."""
        embeder = TDIDFLexicalEmbeder(save_dir=str(tmp_path))

        df = pd.DataFrame({"text": text_data})
        result = embeder.fit_transform(df, ["text"])

        assert result.shape[0] == len(text_data)

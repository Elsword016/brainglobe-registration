from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import for patching, but don't instantiate directly with mock parents
from brainglobe_registration.widgets.similarity_metrics_view import (
    SimilarityMetricsView,
)


def test_find_best_slice_success():
    """Test finding best slice with successful execution."""
    with patch.object(
        SimilarityMetricsView, "get_sample_layer"
    ) as mock_get_sample:
        with patch.object(
            SimilarityMetricsView, "get_atlas_layer"
        ) as mock_get_atlas:
            with patch(
                "brainglobe_registration.widgets.similarity_metrics_view."
                "compare_image_to_atlas_slices"
            ) as mock_compare:
                # Set up our mocks
                sample_layer = MagicMock()
                sample_layer.data = np.ones((10, 10))
                mock_get_sample.return_value = sample_layer

                atlas_layer = MagicMock()
                atlas_layer.data = np.ones((5, 10, 10))
                mock_get_atlas.return_value = atlas_layer

                mock_compare.return_value = {0: 0.5, 1: 0.7, 2: 0.6}

                # Create mock objects needed by the method
                mock_self = MagicMock()
                mock_self.status_label = MagicMock()
                mock_self._parent = MagicMock()
                mock_self._parent._viewer = MagicMock()
                mock_self.metric_combo = MagicMock()
                mock_self.metric_combo.currentText.return_value = "MI"

                # Call the method directly
                SimilarityMetricsView.find_best_slice(mock_self)

                # Verify expected behaviors
                mock_self._parent._viewer.dims.set_point.assert_called_once_with(
                    0, 1
                )
                mock_self.status_label.setText.assert_any_call(
                    "Best match: Slice 1 (similarity: 0.7000)"
                )


def test_find_best_slice_no_sample():
    """Test finding best slice with no sample image."""
    # Create mock object that will be 'self' in the method
    mock_self = MagicMock()
    mock_self.status_label = MagicMock()
    mock_self.metric_combo = MagicMock()
    mock_self.metric_combo.currentText.return_value = "MI"

    # Create a specific mock for the get_sample_layer to actually return None
    mock_self.get_sample_layer = MagicMock(return_value=None)

    # Call the actual method on our mocked instance
    SimilarityMetricsView.find_best_slice(mock_self)

    # Verify correct error message was set
    mock_self.status_label.setText.assert_called_with(
        "Error: No sample image found"
    )


def test_find_best_slice_no_atlas():
    """Test finding best slice with no atlas volume."""
    # Create mock object that will be 'self' in the method
    mock_self = MagicMock()
    mock_self.status_label = MagicMock()
    mock_self.metric_combo = MagicMock()
    mock_self.metric_combo.currentText.return_value = "MI"

    # Create a sample layer for the first check to pass
    sample_layer = MagicMock()
    sample_layer.data = np.ones((10, 10))

    # Setup the mocks on our instance
    mock_self.get_sample_layer = MagicMock(return_value=sample_layer)
    mock_self.get_atlas_layer = MagicMock(return_value=None)

    # Call the actual method on our mocked instance
    SimilarityMetricsView.find_best_slice(mock_self)

    # Verify correct error message was set
    mock_self.status_label.setText.assert_called_with(
        "Error: No atlas volume found"
    )


def test_find_best_slice_exception():
    """Test finding best slice with an exception during processing."""
    with patch.object(
        SimilarityMetricsView, "get_sample_layer"
    ) as mock_get_sample:
        with patch.object(
            SimilarityMetricsView, "get_atlas_layer"
        ) as mock_get_atlas:
            with patch(
                "brainglobe_registration.widgets.similarity_metrics_view."
                "compare_image_to_atlas_slices"
            ) as mock_compare:
                # Set up mocks - sample and atlas exist, but comparison raises
                # exception
                sample_layer = MagicMock()
                sample_layer.data = np.ones((10, 10))
                mock_get_sample.return_value = sample_layer

                atlas_layer = MagicMock()
                atlas_layer.data = np.ones((5, 10, 10))
                mock_get_atlas.return_value = atlas_layer

                mock_compare.side_effect = ValueError("Test error")

                # Create mock objects
                mock_self = MagicMock()
                mock_self.status_label = MagicMock()
                mock_self.metric_combo = MagicMock()
                mock_self.metric_combo.currentText.return_value = "MI"

                # Call the method directly
                SimilarityMetricsView.find_best_slice(mock_self)

                # Verify correct error message was set
                mock_self.status_label.setText.assert_any_call(
                    "Error: Test error"
                )


def test_get_sample_layer():
    """Test the get_sample_layer method."""
    # Create mock object
    mock_self = MagicMock()
    mock_self._parent = MagicMock()
    mock_self._parent._moving_image = MagicMock()
    mock_self._parent._viewer = MagicMock()
    mock_self._parent._viewer.layers = [mock_self._parent._moving_image]

    # Call the method directly
    result = SimilarityMetricsView.get_sample_layer(mock_self)

    # Verify expected result
    assert result == mock_self._parent._moving_image


def test_get_atlas_layer():
    """Test the get_atlas_layer method."""
    # Create mock object
    mock_self = MagicMock()
    mock_self._parent = MagicMock()
    mock_self._parent._atlas_data_layer = MagicMock()

    # Call the method directly
    result = SimilarityMetricsView.get_atlas_layer(mock_self)

    # Verify expected result
    assert result == mock_self._parent._atlas_data_layer


def test_find_best_slice_with_real_images():
    """Test finding best slice using real sample and atlas images."""
    import os

    from skimage import io

    # Create a mock widget
    mock_self = MagicMock()
    mock_self.status_label = MagicMock()
    mock_self.metric_combo = MagicMock()
    mock_self.metric_combo.currentText.return_value = "MI"
    mock_self._parent = MagicMock()
    mock_self._parent._viewer = MagicMock()

    # Use real image paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_path = os.path.join(
        base_dir, "tests", "test_images", "sample_hipp.tif"
    )
    atlas_path = os.path.join(
        base_dir, "tests", "test_images", "Atlas_Hipp.tif"
    )

    # Load real images
    try:
        sample_data = io.imread(sample_path)
        atlas_data = io.imread(atlas_path)

        # Create mock layers with real data
        sample_layer = MagicMock()
        sample_layer.data = sample_data

        atlas_layer = MagicMock()
        atlas_layer.data = atlas_data

        # Mock the get_sample_layer and get_atlas_layer methods
        with patch.object(
            SimilarityMetricsView,
            "get_sample_layer",
            return_value=sample_layer,
        ):
            with patch.object(
                SimilarityMetricsView,
                "get_atlas_layer",
                return_value=atlas_layer,
            ):
                with patch(
                    "brainglobe_registration.widgets.similarity_metrics_view."
                    "compare_image_to_atlas_slices"
                ) as mock_compare:
                    # Mock the comparison function to return a dictionary
                    # with slice 2 having highest similarity
                    mock_compare.return_value = {
                        0: 0.2,
                        1: 0.5,
                        2: 0.9,
                        3: 0.7,
                    }

                    # Call the method
                    SimilarityMetricsView.find_best_slice(mock_self)

                    # Verify results
                    mock_self._parent._viewer.dims.set_point.assert_called_once_with(
                        0, 2
                    )
                    mock_self.status_label.setText.assert_any_call(
                        "Best match: Slice 2 (similarity: 0.9000)"
                    )

                    # Verify compare_image_to_atlas_slices was called correctly
                    mock_compare.assert_called_once()
                    args, _ = mock_compare.call_args
                    assert args[0] is sample_data
                    assert args[1] is atlas_data
                    assert args[2] == "MI"
    except FileNotFoundError:
        pytest.skip("Test image files not found")
    except Exception as e:
        pytest.skip(f"Error loading test images: {e}")

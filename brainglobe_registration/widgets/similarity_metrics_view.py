import napari
import napari.layers
import numpy as np
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..similarity_metrics import compare_image_to_atlas_slices


class SimilarityMetricsView(QWidget):
    """Widget for selecting the best matching atlas slice.

    This widget allows users to select a similarity metric and find the best
    matching atlas slice for a given sample image.

    For now, only 2D registration is supported.
    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # Store reference to the parent widget for accessing viewer and layers
        self._parent = parent

        # Create the layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Create metric selection layout
        metric_select_layout = QHBoxLayout()
        metric_select_layout.addWidget(QLabel("Similarity Metric:"))

        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["NCC", "MI", "SSIM"])
        metric_select_layout.addWidget(self.metric_combo)
        # Add stretch to prevent oversized layout
        metric_select_layout.addStretch()

        # Main button
        self.find_slice_btn = QPushButton("Find Best Matching Slice")
        self.find_slice_btn.clicked.connect(self.find_best_slice)

        # Status label
        self.status_label = QLabel("")

        # Add everything to the layout
        layout.addLayout(metric_select_layout)
        layout.addWidget(self.find_slice_btn)
        layout.addWidget(self.status_label)
        # Add stretch at the end
        layout.addStretch()

    def find_best_slice(self):
        """Find the best matching atlas slice for the sample image.

        This method compares the sample image to each slice of the atlas
        volume using the selected similarity metric and updates the viewer's
        z position to the best matching slice.
        """
        try:
            # Get the current sample image
            sample_layer = self.get_sample_layer()
            if sample_layer is None:
                self.status_label.setText("Error: No sample image found")
                return

            # Get the atlas volume
            atlas_layer = self.get_atlas_layer()
            if atlas_layer is None:
                self.status_label.setText("Error: No atlas volume found")
                return

            # Update status
            self.status_label.setText("Searching for best matching slice...")
            QApplication.processEvents()  # Update UI

            # Get the data - ensure we're working with numpy arrays
            sample_img = np.asarray(sample_layer.data)
            atlas_volume = np.asarray(atlas_layer.data)

            # Get the selected metric name
            metric_name = self.metric_combo.currentText().lower()

            # Compare slices
            results = compare_image_to_atlas_slices(
                sample_img, atlas_volume, metric=metric_name
            )

            if not results:
                self.status_label.setText("Error: No valid results found")
                return

            # Find the best matching slice
            best_slice_idx = max(results.keys(), key=lambda k: results[k])
            best_similarity = results[best_slice_idx]

            # Update the viewer's z position
            if self._parent and hasattr(self._parent, "_viewer"):
                self._parent._viewer.dims.set_point(0, best_slice_idx)

            # Format the similarity value - handle potential array results
            if hasattr(best_similarity, "item"):
                best_similarity_val = float(best_similarity.item())
            else:
                best_similarity_val = float(best_similarity)

            # Update status with the results
            self.status_label.setText(
                f"Best match: Slice {best_slice_idx} "
                f"(similarity: {best_similarity_val:.4f})"
            )

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def get_sample_layer(self):
        """Get the sample image layer.

        If the moving image is available, return it. Otherwise, return any
        image layer that is not the atlas.
        """
        if self._parent and hasattr(self._parent, "_moving_image"):
            if (
                self._parent._moving_image
                and self._parent._moving_image in self._parent._viewer.layers
            ):
                return self._parent._moving_image

            # Fallback to any image layer that's not the atlas
            for layer in self._parent._viewer.layers:
                if (
                    isinstance(layer, napari.layers.Image)
                    and layer != self._parent._atlas_data_layer
                ):
                    return layer

        return None

    def get_atlas_layer(self):
        """Get the atlas volume layer.

        Return the atlas data layer if it exists in the parent widget.
        """
        if self._parent and hasattr(self._parent, "_atlas_data_layer"):
            return self._parent._atlas_data_layer
        return None

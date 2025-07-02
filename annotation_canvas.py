import os
import cv2
import numpy as np
import logging
import traceback
import psutil
from typing import List, Dict, Tuple, Optional
import time

from PyQt5.QtWidgets import (
    QLabel, QMessageBox, QFileDialog, QApplication
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor,
    QKeyEvent, QCursor
)
from PyQt5.QtCore import (
    Qt, QPoint, QRect, QEvent, QSize
)
from PyQt5.QtCore import QTimer
from annotation_io import save_annotation_file, AnnotationFormat, _generate_color_for_class
from image_loader import load_image_folder, get_image_at_index
from annotation_io import AnnotationImporter


class AnnotationCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(100, 100)

        # State variables
        self.image_paths: List[str] = []
        self.undo_stack: List[List[Tuple]] = []
        self.redo_stack: List[List[Tuple]] = []
        self.annotations: Dict[str, List[Tuple]] = {}
        self.current_index: int = -1
        self.original_image: Optional[np.ndarray] = None
        self.display_image: Optional[np.ndarray] = None
        self.scale: float = 1.0
        self.current_boxes: List[Tuple] = []
        self.drawing: bool = False
        self.dragging: bool = False
        self.resizing: bool = False
        self.start_point: QPoint = QPoint()
        self.end_point: QPoint = QPoint()
        self.current_class: str = "Object"
        self.annotation_format: AnnotationFormat = AnnotationFormat.YOLO
        self.annotation_folder: str = ""
        self.last_mouse_pos: Optional[QPoint] = None
        self.selected_box_index: int = -1
        self.hovered_box_index: int = -1
        self.resize_handle: Optional[str] = None
        self.drag_offset: QPoint = QPoint()
        self.class_colors: Dict[str, Tuple[int, int, int]] = {}
        self.default_color: Tuple[int, int, int] = (0, 0, 255)
        self.fit_to_view: bool = True
        self.max_image_size = 50 * 1024 * 1024  # 50MB
        self.max_pixels = 4000 * 4000

        # Drawing-related variables
        self.temp_box = None  # Stores the temporary box during drawing
        self.min_box_size = 10  # Minimum width/height in pixels
        self.temp_box_color = (0, 255, 0)  # Green for temporary box
        self.temp_box_thickness = 1
        self.temp_box_line_type = cv2.LINE_AA  # Anti-aliased lines
        self.crosshair_size = 10  # Size of crosshair marker

        # Memory management
        self.memory_timer = QTimer(self)
        self.memory_timer.timeout.connect(self.check_memory)
        self.memory_timer.start(5000)  # Check every 5 seconds
        if hasattr(parent, 'viewport'):
            parent.viewport().installEventFilter(self)
        self._last_mouse_pos = None

    def eventFilter(self, obj, event):
        """Handle scroll area viewport events for accurate mouse tracking"""
        if obj is self.parent().viewport():
            if event.type() == QEvent.MouseMove:
                self._last_mouse_pos = event.pos()
            elif event.type() == QEvent.Leave:
                self._last_mouse_pos = None
        return super().eventFilter(obj, event)

    def set_annotation_format(self, fmt: AnnotationFormat, folder: str) -> None:
        """Set the annotation format and save folder."""
        self.annotation_format = fmt
        self.annotation_folder = folder
        if not os.path.exists(folder):
            os.makedirs(folder)

    def check_memory(self):
        """Monitor and prevent memory overload"""
        mem = psutil.virtual_memory()
        if mem.available < 200 * 1024 * 1024:  # 200MB left
            self.clear_memory()
            return False
        return True

    def add_image(self, file_path: str) -> bool:
        """Add an image to the annotation set."""
        if not os.path.isfile(file_path):
            return False

        if file_path not in self.image_paths:
            self.image_paths.append(file_path)
            self.annotations[file_path] = []

        index = self.image_paths.index(file_path)
        return self.load_image_by_index(index)

    def safe_imread(self, path: str) -> Optional[np.ndarray]:
        """Safe image loading with multiple fallbacks"""
        try:
            # Check file size first
            if os.path.getsize(path) > self.max_image_size:
                raise MemoryError(f"Image exceeds size limit ({self.max_image_size / 1024 / 1024}MB)")

            img = cv2.imread(path)

            # Proper check for failed image load
            if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                # Try alternative loading methods
                img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
                if img is None or img.size == 0:
                    from PIL import Image
                    pil_img = Image.open(path)
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    if img.size == 0:
                        raise ValueError("All image loading methods failed")

            # Downsample if needed
            h, w = img.shape[:2]
            if h * w > self.max_pixels:
                scale = (self.max_pixels / (h * w)) ** 0.5
                img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
            return img
        except Exception as e:
            logging.error(f"Image load failed: {traceback.format_exc()}")
            return None

    def load_image_by_index(self, index: int) -> bool:
        """Modified to auto-import annotations"""
        if not self.check_memory():
            QMessageBox.warning(self, "Memory Warning", "Low memory - clearing cache")
            self.clear_memory()

        try:
            img_path = self.image_paths[index]
            img = self.safe_imread(img_path)
            if img is None:
                raise ValueError("Failed to load image")

            self.original_image = img
            self.current_boxes = self.annotations.get(img_path, [])

            # Import annotations if none exist yet
            if not self.current_boxes:
                importer = AnnotationImporter(self)
                imported_boxes = importer.import_annotations(img_path)
                if imported_boxes:
                    self.current_boxes = imported_boxes
                    self.annotations[img_path] = imported_boxes

                    # Ensure class colors exist for imported classes
                    for box in imported_boxes:
                        class_name = box[2]
                        if class_name not in self.class_colors:
                            self.class_colors[class_name] = _generate_color_for_class(class_name)

            # Save initial empty state if this is a new image
            if img_path not in self.annotations:
                self._save_state_to_stack()

            self.fit_to_screen()
            self.current_index = index
            self._update_status()
            return True
        except Exception as e:
            error_msg = f"Error loading image: {str(e)}"
            logging.error(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
            return False

    def clear_memory(self):
        """Release resources"""
        self.original_image = None
        if hasattr(self, 'display_image'):
            self.display_image = None
        if hasattr(self, 'cached_pixmaps'):
            self.cached_pixmaps.clear()
        QApplication.processEvents()

    def set_class_label(self, label: str) -> None:
        """Set the current class label for new annotations."""
        self.current_class = label if label.strip() else "Object"
        if self.current_class not in self.class_colors:
            self.class_colors[self.current_class] = _generate_color_for_class(self.current_class)

    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for a class, generating if not exists."""
        if class_name not in self.class_colors:
            self.class_colors[class_name] = _generate_color_for_class(class_name)
        return self.class_colors[class_name]

    def zoom_image(self, factor: float, mouse_pos: Optional[QPoint] = None) -> None:
        """Zoom the image by a factor, centered on mouse position if provided."""
        old_scale = self.scale
        self.scale = max(0.1, min(5.0, self.scale * factor))  # Limit zoom range
        self.fit_to_view = False
        self.update_display()

    def fit_to_screen(self) -> None:
        """Fit the image to the current view size."""
        if self.original_image is None:
            return

        view_size = self.size()
        img_h, img_w = self.original_image.shape[:2]

        width_ratio = view_size.width() / img_w
        height_ratio = view_size.height() / img_h
        self.scale = min(width_ratio, height_ratio)
        self.fit_to_view = True
        self.update_display()

    def _draw_boxes(self, img: np.ndarray) -> np.ndarray:
        """Draw both finalized and temporary boxes with proper styling"""
        img_copy = img.copy()
        handle_size = max(5, int(8 / self.scale))

        # Draw finalized boxes
        for i, box in enumerate(self.current_boxes):
            x1, y1 = box[0].x(), box[0].y()
            x2, y2 = box[1].x(), box[1].y()
            label = box[2] if len(box) > 2 else self.current_class
            color = self._get_class_color(label)

            # Draw box
            thickness = max(1, int(2 / self.scale))
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(img_copy,
                          (x1, y1 - text_size[1] - 4),
                          (x1 + text_size[0], y1),
                          color, -1)
            cv2.putText(img_copy, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw resize handles if selected or hovered
            if i == self.selected_box_index or i == self.hovered_box_index:
                handle_color = (0, 255, 0) if i == self.selected_box_index else (0, 255, 255)
                # Draw corner handles
                cv2.rectangle(img_copy,
                              (x1 - handle_size // 2, y1 - handle_size // 2),
                              (x1 + handle_size // 2, y1 + handle_size // 2),
                              handle_color, -1)
                cv2.rectangle(img_copy,
                              (x2 - handle_size // 2, y1 - handle_size // 2),
                              (x2 + handle_size // 2, y1 + handle_size // 2),
                              handle_color, -1)
                cv2.rectangle(img_copy,
                              (x1 - handle_size // 2, y2 - handle_size // 2),
                              (x1 + handle_size // 2, y2 + handle_size // 2),
                              handle_color, -1)
                cv2.rectangle(img_copy,
                              (x2 - handle_size // 2, y2 - handle_size // 2),
                              (x2 + handle_size // 2, y2 + handle_size // 2),
                              handle_color, -1)

        # Draw temporary box if drawing
        if self.drawing and self.temp_box is not None:
            x1, y1 = self.temp_box[0].x(), self.temp_box[0].y()
            x2, y2 = self.temp_box[1].x(), self.temp_box[1].y()

            # Draw box with dashed line
            line_type = self.temp_box_line_type
            cv2.rectangle(img_copy, (x1, y1), (x2, y2),
                          self.temp_box_color, self.temp_box_thickness, line_type)

            # Draw crosshair at start point
            cv2.drawMarker(img_copy, (x1, y1), self.temp_box_color,
                           cv2.MARKER_CROSS, self.crosshair_size, 1, line_type)

            # Draw size indicator
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            size_text = f"{width}x{height}"
            text_pos = (min(x1, x2), min(y1, y2) - 10)
            cv2.putText(img_copy, size_text, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.temp_box_color, 1, line_type)

        return img_copy

    def update_display(self) -> None:
        """Update the displayed image with current annotations and temporary box"""
        if self.original_image is None:
            self.clear()
            return

        img = self._draw_boxes(self.original_image)

        # Scale and display
        h, w = img.shape[:2]
        new_h, new_w = int(h * self.scale), int(w * self.scale)

        if self.fit_to_view:
            view_size = self.size()
            width_ratio = view_size.width() / w
            height_ratio = view_size.height() / h
            self.scale = min(width_ratio, height_ratio)
            new_h, new_w = int(h * self.scale), int(w * self.scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        self.display_image = resized
        self.setPixmap(self._convert_cv2_to_pixmap(resized))
        self.resize(new_w, new_h)

    def _convert_cv2_to_pixmap(self, cv_img: np.ndarray) -> QPixmap:
        """Convert a NumPy OpenCV image to QPixmap, with validation."""
        if cv_img is None or not isinstance(cv_img, np.ndarray):
            logging.error("Invalid image in _convert_cv2_to_pixmap.")
            return QPixmap()

        if cv_img.size == 0:  # Check for empty array
            logging.error("Empty image array in _convert_cv2_to_pixmap.")
            return QPixmap()
        if cv_img.size == 0:
            return QPixmap()

        if cv_img.ndim == 2:
            # Grayscale image
            height, width = cv_img.shape
            bytes_per_line = width
            q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        elif cv_img.ndim == 3 and cv_img.shape[2] == 3:
            # 3-channel RGB image
            height, width, channel = cv_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        else:
            logging.error(f"Unsupported image shape for QPixmap conversion: {cv_img.shape}")
            return QPixmap()

        return QPixmap.fromImage(q_img)

    def load_images_from_folder(self, folder: str) -> None:
        """Load all images from the specified folder."""
        if self.current_index >= 0 and self.image_paths:
            self.annotations[self.image_paths[self.current_index]] = self.current_boxes.copy()

        self.image_paths = load_image_folder(folder)
        self.current_index = -1
        if self.image_paths:
            self.load_image_by_index(0)



    def _get_box_at_position(self, pos: QPoint) -> Tuple[int, Optional[str]]:
        """Check if position is inside a box or its resize handles."""
        handle_size = max(5, int(8 / self.scale))

        for i, box in enumerate(self.current_boxes):
            x1, y1 = box[0].x(), box[0].y()
            x2, y2 = box[1].x(), box[1].y()
            rect = QRect(QPoint(x1, y1), QPoint(x2, y2))

            # Check if inside box
            if rect.contains(pos):
                # Check resize handles
                if abs(pos.x() - x1) < handle_size and abs(pos.y() - y1) < handle_size:
                    return i, 'nw'  # Northwest handle
                elif abs(pos.x() - x2) < handle_size and abs(pos.y() - y1) < handle_size:
                    return i, 'ne'  # Northeast handle
                elif abs(pos.x() - x1) < handle_size and abs(pos.y() - y2) < handle_size:
                    return i, 'sw'  # Southwest handle
                elif abs(pos.x() - x2) < handle_size and abs(pos.y() - y2) < handle_size:
                    return i, 'se'  # Southeast handle
                return i, None  # Inside box but not on handle
        return -1, None

    def _get_image_coords(self, pos: QPoint) -> QPoint:
        """Convert widget coordinates to image coordinates accounting for zoom and alignment"""
        if self.original_image is None or self.pixmap() is None:
            return QPoint()

        # Get the visible pixmap area
        pixmap = self.pixmap()
        if pixmap is None:
            return QPoint()

        # Calculate content margins (for centered images)
        widget_size = self.size()
        pixmap_size = pixmap.size()
        left_margin = max(0, (widget_size.width() - pixmap_size.width()) // 2)
        top_margin = max(0, (widget_size.height() - pixmap_size.height()) // 2)

        # Adjust for scroll if using QScrollArea
        if hasattr(self.parent(), 'viewport'):
            scroll_pos = self.parent().viewport().mapFromGlobal(self.mapToGlobal(pos))
            pos = scroll_pos

        # Convert to image coordinates
        x = int((pos.x() - left_margin) / self.scale)
        y = int((pos.y() - top_margin) / self.scale)

        # Clamp to image dimensions
        img_w = self.original_image.shape[1]
        img_h = self.original_image.shape[0]
        return QPoint(
            max(0, min(x, img_w - 1)),
            max(0, min(y, img_h - 1)))

    def mousePressEvent(self, event):
        if self.original_image is None:
            return

        pos = self._get_image_coords(event.pos())

        # Check if clicking on a box handle
        self.selected_box_index, self.resize_handle = self._get_box_at_position(pos)

        if self.selected_box_index >= 0:
            if event.button() == Qt.LeftButton:
                if self.resize_handle:
                    self.resizing = True
                    self.start_point = pos
                else:
                    # Move existing box
                    self.dragging = True
                    self.drag_offset = pos - self.current_boxes[self.selected_box_index][0]
            elif event.button() == Qt.RightButton:
                # Delete box on right-click
                self._save_state_to_stack()
                del self.current_boxes[self.selected_box_index]
                self.selected_box_index = -1
                self.update_display()
        else:
            # Start drawing new box
            if event.button() == Qt.LeftButton:
                self.drawing = True
                self.start_point = pos
                self.temp_box = (pos, pos, self.current_class)  # Initialize temp box
                self.update_display()

    def mouseMoveEvent(self, event):
        pos = self._get_image_coords(event.pos())
        self.last_mouse_pos = event.pos()

        if self.drawing:
            # Update temporary box during drawing
            self.temp_box = (self.start_point, pos, self.current_class)

            # Calculate and show size while drawing
            width = abs(pos.x() - self.start_point.x())
            height = abs(pos.y() - self.start_point.y())
            self._update_status(f"Drawing: {width}x{height} px")

            self.update_display()

        elif self.dragging and self.selected_box_index >= 0:
            # Move the entire box
            box = list(self.current_boxes[self.selected_box_index])
            offset = pos - self.drag_offset

            # Calculate new position while maintaining box dimensions
            width = box[1].x() - box[0].x()
            height = box[1].y() - box[0].y()

            # Constrain to image boundaries
            img_width = self.original_image.shape[1]
            img_height = self.original_image.shape[0]
            new_x1 = max(0, min(offset.x(), img_width - width))
            new_y1 = max(0, min(offset.y(), img_height - height))

            box[0] = QPoint(new_x1, new_y1)
            box[1] = QPoint(new_x1 + width, new_y1 + height)
            self.current_boxes[self.selected_box_index] = tuple(box)

            self._update_status(f"Moving: {width}x{height} px")
            self.update_display()

        elif self.resizing and self.selected_box_index >= 0:
            # Resize the box using the appropriate handle
            box = list(self.current_boxes[self.selected_box_index])
            x1, y1 = box[0].x(), box[0].y()
            x2, y2 = box[1].x(), box[1].y()

            # Constrain resizing to image boundaries
            img_width = self.original_image.shape[1]
            img_height = self.original_image.shape[0]

            if self.resize_handle == 'nw':  # Northwest handle
                new_x1 = min(pos.x(), x2 - self.min_box_size)
                new_y1 = min(pos.y(), y2 - self.min_box_size)
                box[0] = QPoint(max(0, new_x1), max(0, new_y1))
            elif self.resize_handle == 'ne':  # Northeast handle
                new_x2 = max(pos.x(), x1 + self.min_box_size)
                new_y1 = min(pos.y(), y2 - self.min_box_size)
                box[1].setX(min(img_width, new_x2))
                box[0].setY(max(0, new_y1))
            elif self.resize_handle == 'sw':  # Southwest handle
                new_x1 = min(pos.x(), x2 - self.min_box_size)
                new_y2 = max(pos.y(), y1 + self.min_box_size)
                box[0].setX(max(0, new_x1))
                box[1].setY(min(img_height, new_y2))
            elif self.resize_handle == 'se':  # Southeast handle
                new_x2 = max(pos.x(), x1 + self.min_box_size)
                new_y2 = max(pos.y(), y1 + self.min_box_size)
                box[1] = QPoint(min(img_width, new_x2), min(img_height, new_y2))

            self.current_boxes[self.selected_box_index] = tuple(box)

            # Update status with new size
            width = box[1].x() - box[0].x()
            height = box[1].y() - box[0].y()
            self._update_status(f"Resizing: {width}x{height} px")
            self.update_display()

        else:
            # Highlight hovered box and update cursor
            self.hovered_box_index, handle = self._get_box_at_position(pos)

            # Set appropriate cursor
            if handle == 'nw' or handle == 'se':
                self.setCursor(Qt.SizeFDiagCursor)
            elif handle == 'ne' or handle == 'sw':
                self.setCursor(Qt.SizeBDiagCursor)
            elif self.hovered_box_index >= 0:
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

            self.update_display()

    def mouseReleaseEvent(self, event):
        if self.drawing and event.button() == Qt.LeftButton:
            self.drawing = False
            end_point = self._get_image_coords(event.pos())

            # Calculate final box coordinates
            x1, y1 = self.start_point.x(), self.start_point.y()
            x2, y2 = end_point.x(), end_point.y()
            width = abs(x2 - x1)
            height = abs(y2 - y1)

            # Only add if meets minimum size
            if width >= self.min_box_size and height >= self.min_box_size:
                # Save state before adding if this is the first box
                if not self.current_boxes:
                    self._save_state_to_stack()

                final_box = (
                    QPoint(min(x1, x2), min(y1, y2)),
                    QPoint(max(x1, x2), max(y1, y2)),
                    self.current_class
                )
                self.current_boxes.append(final_box)
                self._save_state_to_stack()
                self.annotations[self.image_paths[self.current_index]] = self.current_boxes.copy()
            self.temp_box = None
            self._update_status()
            self.update_display()

        elif self.dragging or self.resizing:
            self._save_state_to_stack()
            self.dragging = False
            self.resizing = False
            self._update_status()

    def _save_state_to_stack(self):
        """Debounce rapid saves during drag/resize"""
        if not hasattr(self, '_last_save_time'):
            self._last_save_time = 0

        now = time.time()
        if now - self._last_save_time > 0.5:  # 500ms throttle
            if self.current_index >= 0:
                img_path = self.image_paths[self.current_index]
                if not self.undo_stack or self.undo_stack[-1][1] != self.current_boxes:
                    self.undo_stack.append((img_path, self.current_boxes.copy()))
                    self.redo_stack.clear()
            self._last_save_time = now

    def _update_status(self, message: str = ""):
        """Update status bar message."""
        if hasattr(self.parent(), 'statusBar'):
            if message:
                self.parent().statusBar().showMessage(message)
            else:
                self.parent().statusBar().showMessage(
                    f"Image {self.current_index + 1}/{len(self.image_paths)} | "
                    f"Zoom: {self.scale:.1f}x | "
                    f"Boxes: {len(self.current_boxes)}"
                )

    def undo(self) -> None:
        """Undo the last annotation action."""
        if self.undo_stack and self.current_index >= 0:
            current_img = self.image_paths[self.current_index]
            self.redo_stack.append((current_img, self.current_boxes.copy()))

            img_path, boxes = self.undo_stack.pop()
            self.current_boxes = boxes.copy()
            self.annotations[img_path] = boxes.copy()

            # If we undid to a different image, load it
            if img_path != current_img and img_path in self.image_paths:
                idx = self.image_paths.index(img_path)
                self.load_image_by_index(idx)
            else:
                self.update_display()
                self._update_status()

    def redo(self) -> None:
        """Redo the last undone action."""
        if self.redo_stack and self.current_index >= 0:
            current_img = self.image_paths[self.current_index]
            self.undo_stack.append((current_img, self.current_boxes.copy()))

            img_path, boxes = self.redo_stack.pop()
            self.current_boxes = boxes.copy()
            self.annotations[img_path] = boxes.copy()

            # If we redid to a different image, load it
            if img_path != current_img and img_path in self.image_paths:
                idx = self.image_paths.index(img_path)
                self.load_image_by_index(idx)
            else:
                self.update_display()
                self._update_status()

    def save_annotations(self) -> None:
        """Save all annotations to disk with file dialog."""
        if not self.image_paths:
            QMessageBox.warning(self, "Error", "No images loaded to save annotations for")
            return

        # Prompt for save location if not set
        if not self.annotation_folder:
            folder = QFileDialog.getExistingDirectory(
                self, "Select Folder to Save Annotations")
            if not folder:
                return
            self.annotation_folder = folder

        try:
            coco_data = None
            success_count = 0

            for img_path, boxes in self.annotations.items():
                if not boxes:
                    continue

                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        logging.error(f"Failed to read image for annotation: {img_path}")
                        continue

                    if self.annotation_format == AnnotationFormat.COCO:
                        coco_data = save_annotation_file(
                            img_path, boxes, self.annotation_format,
                            self.annotation_folder, img.shape, coco_data,
                        )
                    else:
                        save_annotation_file(
                            img_path, boxes, self.annotation_format,
                            self.annotation_folder, img.shape,

                        )
                    success_count += 1
                except Exception as e:
                    logging.error(f"Failed to save annotations for {img_path}: {e}")

            msg = f"Successfully saved {success_count} annotations"
            if success_count < len(self.annotations):
                msg += f" (failed {len(self.annotations) - success_count})"
            QMessageBox.information(self, "Save Complete", msg)

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save annotations: {str(e)}")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle keyboard shortcuts."""
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_Z:
                self.undo()
            elif event.key() == Qt.Key_Y:
                self.redo()
            elif event.key() == Qt.Key_S:
                self.save_annotations()
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            self.zoom_image(1.25, self.last_mouse_pos)
        elif event.key() == Qt.Key_Minus:
            self.zoom_image(0.8, self.last_mouse_pos)
        elif event.key() == Qt.Key_F:
            self.fit_to_screen()
        elif event.key() == Qt.Key_Delete and self.selected_box_index >= 0:
            self._save_state_to_stack()
            del self.current_boxes[self.selected_box_index]
            self.selected_box_index = -1
            self.update_display()

    def next_image(self) -> None:
        """Load the next image in the sequence."""
        if self.image_paths and self.current_index < len(self.image_paths) - 1:
            self.load_image_by_index(self.current_index + 1)

    def prev_image(self) -> None:
        """Load the previous image in the sequence."""
        if self.image_paths and self.current_index > 0:
            self.load_image_by_index(self.current_index - 1)

    def resizeEvent(self, event):
        if not hasattr(self, '_resizing'):
            self._resizing = False

        if self._resizing:
            return

        self._resizing = True
        try:
            if self.fit_to_view and self.original_image is not None:
                self.fit_to_screen()
        finally:
            self._resizing = False

        super().resizeEvent(event)

import os
from enum import Enum
from typing import Optional, Dict, Tuple
import psutil
import traceback
import logging

from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QPushButton, QListWidget,
    QLabel, QScrollArea, QFileDialog, QFrame, QSplitter,
    QLineEdit, QComboBox, QStatusBar, QShortcut, QMenu,
    QColorDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QKeySequence, QIcon, QColor
from PyQt5.QtWidgets import QApplication

from annotation_canvas import AnnotationCanvas
from annotation_io import AnnotationFormat
from image_loader import load_image_folder, add_image_to_list


class AnnotatorMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Annotation Tool")
        self.setMinimumSize(1200, 700)

        # Initialize UI components
        self._setup_ui()
        self._setup_shortcuts()

        # State variables
        self.annotation_format = AnnotationFormat.YOLO
        self.annotation_folder = ""
        self.coco_data = None  # For COCO format tracking
        self.min_memory_mb = 500

    def _setup_ui(self):
        """Initialize all UI components."""
        # Apply stylesheet
        self.setStyleSheet(self._get_stylesheet())

        # Create main canvas
        self.canvas = AnnotationCanvas(self)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # Left Panel (Class Bin + Tools)
        left_frame = QFrame()
        left_frame.setMinimumWidth(250)
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # Class Bin Section
        left_layout.addWidget(QLabel("Class Bin", self))
        self.class_list_widget = QListWidget(self)
        self.class_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.class_list_widget.customContextMenuRequested.connect(self._show_class_context_menu)
        self.class_list_widget.itemClicked.connect(self._select_class_from_bin)
        left_layout.addWidget(self.class_list_widget)

        # Class Input
        class_input_layout = QVBoxLayout()
        self.class_input = QLineEdit(self)
        self.class_input.setPlaceholderText("Add new class")
        self.class_input.returnPressed.connect(self._add_class_to_bin)
        class_input_layout.addWidget(self.class_input)

        add_class_btn = QPushButton("Add Class", self)
        add_class_btn.clicked.connect(self._add_class_to_bin)
        class_input_layout.addWidget(add_class_btn)
        left_layout.addLayout(class_input_layout)

        # Tools Section
        left_layout.addWidget(QLabel("Tools", self))

        # Annotation Format
        self.format_combo = QComboBox(self)
        self.format_combo.addItems([fmt.name for fmt in AnnotationFormat])
        self.format_combo.currentIndexChanged.connect(self._select_annotation_format)
        left_layout.addWidget(self.format_combo)

        # Navigation Buttons
        nav_btn_layout = QVBoxLayout()
        prev_btn = QPushButton("Previous Image (←)", self)
        prev_btn.clicked.connect(self.canvas.prev_image)
        next_btn = QPushButton("Next Image (→)", self)
        next_btn.clicked.connect(self.canvas.next_image)
        nav_btn_layout.addWidget(prev_btn)
        nav_btn_layout.addWidget(next_btn)
        left_layout.addLayout(nav_btn_layout)

        # Editing Tools
        undo_btn = QPushButton("Undo (Ctrl+Z)", self)
        undo_btn.clicked.connect(self.canvas.undo)
        redo_btn = QPushButton("Redo (Ctrl+Y)", self)
        redo_btn.clicked.connect(self.canvas.redo)
        left_layout.addWidget(undo_btn)
        left_layout.addWidget(redo_btn)

        # Zoom Controls
        zoom_in_btn = QPushButton("Zoom In (+)", self)
        zoom_in_btn.clicked.connect(lambda: self.canvas.zoom_image(1.25))
        zoom_out_btn = QPushButton("Zoom Out (-)", self)
        zoom_out_btn.clicked.connect(lambda: self.canvas.zoom_image(0.8))
        fit_btn = QPushButton("Fit to View (F)", self)
        fit_btn.clicked.connect(self.canvas.fit_to_screen)
        left_layout.addWidget(zoom_in_btn)
        left_layout.addWidget(zoom_out_btn)
        left_layout.addWidget(fit_btn)

        # Folder Operations
        select_folder_btn = QPushButton("Select Image Folder", self)
        select_folder_btn.clicked.connect(self._load_folder)
        left_layout.addWidget(select_folder_btn)

        save_btn = QPushButton("Save Annotations (Ctrl+S)", self)
        save_btn.clicked.connect(self.canvas.save_annotations)
        left_layout.addWidget(save_btn)

        left_frame.setLayout(left_layout)
        splitter.addWidget(left_frame)

        # Center Panel (Canvas)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.canvas)
        splitter.addWidget(scroll_area)

        # Right Panel (Image List)
        right_frame = QFrame()
        right_frame.setMinimumWidth(200)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(5)
        right_layout.setContentsMargins(5, 5, 5, 5)

        self.image_list = QListWidget(self)
        self.image_list.itemClicked.connect(self._load_image_from_list)
        right_layout.addWidget(QLabel("Image Files", self))
        right_layout.addWidget(self.image_list)

        add_img_btn = QPushButton("Add Image", self)
        add_img_btn.clicked.connect(self._add_image)
        right_layout.addWidget(add_img_btn)

        right_frame.setLayout(right_layout)
        splitter.addWidget(right_frame)

        # Set initial splitter sizes
        splitter.setSizes([250, 800, 200])

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._update_status("Ready")

    def _setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        QShortcut(QKeySequence("Ctrl+Z"), self, self.canvas.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.canvas.redo)
        QShortcut(QKeySequence("Ctrl+S"), self, self.canvas.save_annotations)
        QShortcut(QKeySequence("+"), self, lambda: self.canvas.zoom_image(1.25))
        QShortcut(QKeySequence("-"), self, lambda: self.canvas.zoom_image(0.8))
        QShortcut(QKeySequence("F"), self, self.canvas.fit_to_screen)
        QShortcut(QKeySequence("Right"), self, self.canvas.next_image)
        QShortcut(QKeySequence("Left"), self, self.canvas.prev_image)
        QShortcut(QKeySequence("Delete"), self, self._delete_selected_box)

    def _get_stylesheet(self) -> str:
        """Return the Qt stylesheet for the application."""
        return """
        QMainWindow {
            background-color: #f0f0f0;
        }
        QFrame {
            background-color: white;
            border-radius: 5px;
            padding: 5px;
        }
        QPushButton {
            background-color: #e0e0e0;
            border: 1px solid #c0c0c0;
            border-radius: 4px;
            padding: 5px;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #d0d0d0;
        }
        QPushButton:pressed {
            background-color: #b0b0b0;
        }
        QListWidget {
            border: 1px solid #c0c0c0;
            border-radius: 4px;
            background-color: white;
        }
        QLineEdit {
            border: 1px solid #c0c0c0;
            border-radius: 4px;
            padding: 3px;
        }
        QComboBox {
            border: 1px solid #c0c0c0;
            border-radius: 4px;
            padding: 3px;
        }
        QLabel {
            font-weight: bold;
        }
        QStatusBar {
            background-color: #e0e0e0;
            border-top: 1px solid #c0c0c0;
        }
        """

    def _show_class_context_menu(self, pos):
        """Show context menu for class items."""
        item = self.class_list_widget.itemAt(pos)
        if not item:
            return

        menu = QMenu()
        change_color = menu.addAction("Change Color")
        delete_class = menu.addAction("Delete Class")

        action = menu.exec_(self.class_list_widget.mapToGlobal(pos))
        if action == change_color:
            self._change_class_color(item)
        elif action == delete_class:
            self._delete_class(item)

    def _change_class_color(self, item):
        """Change the color for a class."""
        class_name = item.text()
        color = QColorDialog.getColor(initial=QColor(*self.canvas._get_class_color(class_name)))
        if color.isValid():
            self.canvas.class_colors[class_name] = (color.red(), color.green(), color.blue())
            self.canvas.update_display()

    def _delete_class(self, item):
        """Delete a class from the list."""
        class_name = item.text()
        reply = QMessageBox.question(
            self, 'Delete Class',
            f"Delete class '{class_name}'? This won't remove existing annotations.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.class_list_widget.takeItem(self.class_list_widget.row(item))
            if class_name in self.canvas.class_colors:
                del self.canvas.class_colors[class_name]

    def _delete_selected_box(self):
        """Delete the currently selected box."""
        if self.canvas.selected_box_index >= 0:
            self.canvas._save_state_to_stack()
            del self.canvas.current_boxes[self.canvas.selected_box_index]
            self.canvas.selected_box_index = -1
            self.canvas.update_display()

    def _update_status(self, message: str = ""):
        """Update status bar message."""
        if message:
            self.status_bar.showMessage(message)
        elif hasattr(self.canvas, 'current_index'):
            self.status_bar.showMessage(
                f"Image {self.canvas.current_index + 1}/{len(self.canvas.image_paths)} | "
                f"Zoom: {self.canvas.scale:.1f}x | "
                f"Boxes: {len(self.canvas.current_boxes)}"
            )

    def _select_annotation_format(self):
        """Handle annotation format selection."""
        fmt_name = self.format_combo.currentText()
        try:
            self.annotation_format = AnnotationFormat[fmt_name]
            folder = QFileDialog.getExistingDirectory(
                self, "Select Folder to Save Annotations")
            if folder:
                self.annotation_folder = folder
                self.canvas.set_annotation_format(self.annotation_format, folder)
                self._update_status(f"Format set to {fmt_name}, saving to {folder}")
        except KeyError:
            self._update_status(f"Unknown format: {fmt_name}")

    def _load_folder(self):
        """Load images from selected folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.canvas.load_images_from_folder(folder)
            self.image_list.clear()
            for path in self.canvas.image_paths:
                self.image_list.addItem(os.path.basename(path))

            if self.canvas.image_paths:
                self._update_status(f"Loaded {len(self.canvas.image_paths)} images from {folder}")
            else:
                self._update_status(f"No images found in {folder}")

    def _load_image_from_list(self, item):
        """Load image selected from the list."""
        index = self.image_list.row(item)
        self.canvas.load_image_by_index(index)
        self._update_status()

    def _add_image(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Add Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)")

        if not file:
            return

        try:
            # Check memory first
            mem = psutil.virtual_memory()
            if mem.available < self.min_memory_mb * 1024 * 1024:
                raise MemoryError(
                    f"Only {mem.available / 1024 / 1024:.0f}MB available, "
                    f"need {self.min_memory_mb}MB"
                )

            # Show loading state
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.status_bar.showMessage("Loading image...")
            QApplication.processEvents()

            # Use safe loading
            if not self.canvas.add_image(file):
                raise ValueError("Failed to add image")

            self.image_list.addItem(os.path.basename(file))
            self._update_status(f"Loaded: {os.path.basename(file)}")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Couldn't load image:\n{str(e)}\n\n"
                f"{traceback.format_exc() if isinstance(e, MemoryError) else ''}"
            )
            logging.error(f"Image load failed: {traceback.format_exc()}")
        finally:
            QApplication.restoreOverrideCursor()

    def _add_class_to_bin(self):
        """Add new class to the class bin."""
        new_class = self.class_input.text().strip()
        if new_class:
            # Check for duplicates
            existing_classes = [
                self.class_list_widget.item(i).text()
                for i in range(self.class_list_widget.count())
            ]

            if new_class not in existing_classes:
                self.class_list_widget.addItem(new_class)
                self.class_input.clear()
                self.canvas.set_class_label(new_class)
                self._update_status(f"Added class: {new_class}")
            else:
                self._update_status(f"Class '{new_class}' already exists")

    def _select_class_from_bin(self, item):
        """Select class from the class bin."""
        selected_label = item.text()
        self.canvas.set_class_label(selected_label)

        # Highlight selection
        for i in range(self.class_list_widget.count()):
            self.class_list_widget.item(i).setBackground(Qt.white)
        item.setBackground(QColor(200, 230, 255))  # Light blue highlight

        self._update_status(f"Selected class: {selected_label}")
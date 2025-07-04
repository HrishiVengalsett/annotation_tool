import json
import os
import subprocess
import sys
from enum import Enum
from typing import Optional, Dict, Tuple

import cv2
import psutil
import traceback
import logging

from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QListWidget,
    QLabel, QScrollArea, QFileDialog, QFrame, QSplitter, QGroupBox,
    QLineEdit, QComboBox, QStatusBar, QShortcut, QMenu,
    QColorDialog, QMessageBox, QDialog, QSpinBox, QDialogButtonBox, QWidget, QCheckBox
)
from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QKeySequence, QIcon, QColor, QImage, QPixmap
from PyQt5.QtWidgets import QApplication

from annotation_canvas import AnnotationCanvas
from annotation_io import AnnotationFormat, DatasetExporter, ImageAugmenter, save_annotation_file, \
    save_visual_annotation_image
from image_loader import load_image_folder, add_image_to_list


class AugmentationPreviewDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Augmentations to Keep")
        self.setMinimumSize(800, 600)
        self.selected_items = []

        # Main layout
        self.layout = QVBoxLayout()

        # Scroll area for images
        self.scroll_area = QScrollArea()
        self.scroll_content = QWidget()
        self.scroll_layout = QGridLayout()
        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_content)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

        # Buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

        self.setLayout(self.layout)
        self.row = 0
        self.col = 0

    def add_image(self, name, img, boxes, class_colors):
        """Add an augmented image preview to the dialog"""
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout()

        # Create checkbox
        checkbox = QCheckBox(name)
        checkbox.setChecked(True)

        # Convert image to QPixmap
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)

        # Create thumbnail label
        label = QLabel()
        label.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio))
        label.setAlignment(Qt.AlignCenter)

        # Draw boxes on a copy for visualization
        vis_img = img.copy()
        for box in boxes:
            p1, p2, class_name = box
            color = class_colors.get(class_name, (0, 0, 255))
            cv2.rectangle(vis_img,
                          (p1.x(), p1.y()),
                          (p2.x(), p2.y()),
                          color, 2)

        # Add to layout
        layout.addWidget(checkbox)
        layout.addWidget(label)
        frame.setLayout(layout)

        # Store the image data with the checkbox
        frame.img_data = (name, img, boxes)
        checkbox.toggled.connect(lambda state, f=frame: self._toggle_selection(f, state))

        # Add to grid layout
        self.scroll_layout.addWidget(frame, self.row, self.col)
        self.col += 1
        if self.col > 2:  # 3 columns
            self.col = 0
            self.row += 1

    def _toggle_selection(self, frame, state):
        """Add/remove image from selected items based on checkbox"""
        if state:
            self.selected_items.append(frame.img_data)
        else:
            self.selected_items.remove(frame.img_data)

    def get_selected_items(self):
        """Return list of (name, image, boxes) tuples"""
        return self.selected_items


class AnnotatorMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Annotation Tool")
        self.setMinimumSize(1200, 700)

        self.annotation_format = AnnotationFormat.YOLO
        self.annotation_folder = ""
        self.coco_data = None
        self.min_memory_mb = 500

        self._setup_ui()
        self._setup_shortcuts()

    def _setup_ui(self):
        self.setStyleSheet(self._get_stylesheet())
        self.canvas = AnnotationCanvas(self)
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        left_frame = QFrame()
        left_frame.setMinimumWidth(250)
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # Class Management Section
        class_group = QGroupBox("Class Management")
        class_layout = QVBoxLayout()
        class_layout.setSpacing(5)

        self.class_list_widget = QListWidget(self)
        self.class_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.class_list_widget.customContextMenuRequested.connect(self._show_class_context_menu)
        self.class_list_widget.itemClicked.connect(self._select_class_from_bin)
        class_layout.addWidget(self.class_list_widget)

        class_input_layout = QHBoxLayout()
        self.class_input = QLineEdit(self)
        self.class_input.setPlaceholderText("New class name")
        self.class_input.returnPressed.connect(self._add_class_to_bin)
        class_input_layout.addWidget(self.class_input)

        add_class_btn = QPushButton("Add", self)
        add_class_btn.clicked.connect(self._add_class_to_bin)
        class_input_layout.addWidget(add_class_btn)
        class_layout.addLayout(class_input_layout)

        class_group.setLayout(class_layout)
        left_layout.addWidget(class_group)

        # Image Navigation Section
        nav_group = QGroupBox("Image Navigation")
        nav_layout = QGridLayout()
        nav_layout.setSpacing(5)

        prev_btn = QPushButton("Previous (←)", self)
        prev_btn.clicked.connect(self.canvas.prev_image)
        nav_layout.addWidget(prev_btn, 0, 0)

        next_btn = QPushButton("Next (→)", self)
        next_btn.clicked.connect(self.canvas.next_image)
        nav_layout.addWidget(next_btn, 0, 1)

        zoom_in_btn = QPushButton("Zoom In (+)", self)
        zoom_in_btn.clicked.connect(lambda: self.canvas.zoom_image(1.25))
        nav_layout.addWidget(zoom_in_btn, 1, 0)

        zoom_out_btn = QPushButton("Zoom Out (-)", self)
        zoom_out_btn.clicked.connect(lambda: self.canvas.zoom_image(0.8))
        nav_layout.addWidget(zoom_out_btn, 1, 1)

        undo_btn = QPushButton("Undo (Ctrl+Z)", self)
        undo_btn.clicked.connect(self.canvas.undo)
        nav_layout.addWidget(undo_btn, 2, 0)

        redo_btn = QPushButton("Redo (Ctrl+Y)", self)
        redo_btn.clicked.connect(self.canvas.redo)
        nav_layout.addWidget(redo_btn, 2, 1)

        delete_box_btn = QPushButton("Delete Box (Del)", self)
        delete_box_btn.clicked.connect(self._delete_selected_box)
        nav_layout.addWidget(delete_box_btn, 3, 0, 1, 2)

        nav_group.setLayout(nav_layout)
        left_layout.addWidget(nav_group)

        # File Operations Section
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()
        file_layout.setSpacing(5)



        self.format_combo = QComboBox(self)
        self.format_combo.addItems([f.name for f in AnnotationFormat])
        self.format_combo.currentIndexChanged.connect(self._select_annotation_format)
        file_layout.addWidget(self.format_combo)

        select_folder_btn = QPushButton("Select Image Folder", self)
        select_folder_btn.clicked.connect(self._load_folder)
        file_layout.addWidget(select_folder_btn)

        save_btn = QPushButton("Save Annotations (Ctrl+S)", self)
        save_btn.clicked.connect(self.canvas.save_annotations)
        file_layout.addWidget(save_btn)

        export_btn = QPushButton("Export Dataset", self)
        export_btn.clicked.connect(self._export_dataset)
        file_layout.addWidget(export_btn)

        augment_btn = QPushButton("Create Augmentations", self)
        augment_btn.clicked.connect(self._handle_augmentations)
        file_layout.addWidget(augment_btn)

        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        left_layout.addStretch()
        left_frame.setLayout(left_layout)
        splitter.addWidget(left_frame)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.canvas)
        splitter.addWidget(scroll_area)

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

        splitter.setSizes([250, 800, 200])
        self._setup_menubar()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._update_status("Ready")

    def _setup_menubar(self):
        """Initialize the main menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        # Add COCO dataset action
        load_coco_action = file_menu.addAction("Load COCO Dataset")
        load_coco_action.triggered.connect(self.load_coco_dataset)
        load_coco_action.setShortcut("Ctrl+O")

        # Add separator
        file_menu.addSeparator()

        # Add exit action
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut("Ctrl+Q")

    def _export_dataset(self):
        """Handle the dataset export functionality"""
        # Initialize exporter if it doesn't exist
        if not hasattr(self, 'exporter'):
            from annotation_io import DatasetExporter
            self.exporter = DatasetExporter()

        # Check if there are any annotations to export
        if not hasattr(self.canvas, 'annotations') or not self.canvas.annotations:
            QMessageBox.warning(self, "Error", "No annotations to export")
            return

        # Perform the export
        success = self.exporter.export_dataset(self.canvas, self)

        # Provide feedback
        if success:
            self._update_status("Dataset exported successfully")
        else:
            self._update_status("Dataset export failed")

    def _handle_augmentations(self):
        if not self.canvas.image_paths:
            QMessageBox.warning(self, "Error", "No images loaded to augment")
            return

        dialog = AugmentationDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            n = dialog.get_augmentation_count()
            self._generate_augmentations(n)

    def _generate_augmentations(self, n_per_image):
        """Generate augmented versions of all loaded images"""
        try:
            # Initialize augmenter
            self.augmenter = ImageAugmenter()

            # Create preview dialog
            preview_dialog = AugmentationPreviewDialog(self)

            # Process each image
            for img_path in self.canvas.image_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                boxes = self.canvas.annotations.get(img_path, [])
                if not boxes:
                    continue

                # Generate augmentations
                augmented = self.augmenter.augment_image(img, boxes, n_per_image)

                # Add to preview dialog
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                for i, (aug_img, aug_boxes) in enumerate(augmented):
                    preview_dialog.add_image(
                        f"{base_name}_aug{i + 1}",
                        aug_img,
                        aug_boxes,
                        self.canvas.class_colors
                    )

            # Show preview and get selection
            if preview_dialog.exec_() == QDialog.Accepted:
                selected = preview_dialog.get_selected_items()
                if selected:
                    self._save_augmentations(selected)
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Saved {len(selected)} augmented images"
                    )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Augmentation failed: {str(e)}")
            logging.error(f"Augmentation error: {traceback.format_exc()}")

    def _save_augmentations(self, selected_items):
        """Save augmentations to unified folder structure"""
        try:
            # 1. Create base folder
            base_dir = os.path.join(os.path.dirname(self.canvas.image_paths[0]), "augmented_images")
            os.makedirs(base_dir, exist_ok=True)

            # 2. Create subfolders
            folders = {
                'images': os.path.join(base_dir, "images/train"),
                'annotations': {
                    'yolo': os.path.join(base_dir, "annotations/yolo"),
                    'coco': os.path.join(base_dir, "annotations/coco"),
                    'custom': os.path.join(base_dir, "annotations/custom_txt")
                },
                'visual': os.path.join(base_dir, "visual_annotations")
            }

            # Create all directories
            for folder in [folders['images'], *folders['annotations'].values(), folders['visual']]:
                os.makedirs(folder, exist_ok=True)

            # 3. Save each augmentation
            success_count = 0
            for name, img, boxes in selected_items:
                try:
                    # Save image
                    img_name = f"{name}.jpg"
                    img_path = os.path.join(folders['images'], img_name)
                    if not cv2.imwrite(img_path, img):
                        raise IOError(f"Failed to write {img_path}")

                    # Save annotations
                    img_shape = img.shape
                    for fmt, path in folders['annotations'].items():
                        save_annotation_file(
                            img_path,
                            boxes,
                            AnnotationFormat[fmt.upper()],
                            path,
                            img_shape
                        )

                    # Save visual
                    save_visual_annotation_image(
                        img_path,
                        boxes,
                        folders['visual'],
                        self.canvas.class_colors
                    )

                    success_count += 1
                except Exception as e:
                    logging.error(f"Failed to save {name}: {str(e)}")
                    continue

            # 4. Verify output
            if success_count > 0:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Saved {success_count} augmentations to:\n{base_dir}"
                )
                # Open folder in file explorer
                if sys.platform == "win32":
                    os.startfile(base_dir)
                elif sys.platform == "darwin":
                    subprocess.run(["open", base_dir])
                else:
                    subprocess.run(["xdg-open", base_dir])
            else:
                QMessageBox.warning(self, "Error", "No augmentations were saved")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Critical Error",
                f"Could not create output structure:\n{str(e)}"
            )
            logging.critical(f"Augmentation save failed: {traceback.format_exc()}")

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
        QGroupBox {
            border: 1px solid #c0c0c0;
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 15px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px;
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

    def load_coco_dataset(self):
        """Load a COCO format annotation file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open COCO Dataset", "", "JSON Files (*.json)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                self.canvas.coco_data = json.load(f)

            # Load images from COCO dataset
            image_dir = os.path.dirname(file_path)
            self.canvas.image_paths = [
                os.path.join(image_dir, img['file_name'])
                for img in self.canvas.coco_data['images']
                if os.path.exists(os.path.join(image_dir, img['file_name']))
            ]

            if self.canvas.image_paths:
                self.canvas.load_image_by_index(0)
                self._update_status(f"Loaded COCO dataset with {len(self.canvas.image_paths)} images")
            else:
                QMessageBox.warning(self, "Error", "No images found in COCO dataset directory")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load COCO dataset: {str(e)}")


class AugmentationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Augmentation")
        layout = QVBoxLayout()

        self.label = QLabel("How many augmented images would you like to generate per original image?")
        layout.addWidget(self.label)

        self.spin_box = QSpinBox()
        self.spin_box.setRange(3, 15)
        self.spin_box.setValue(5)
        layout.addWidget(self.spin_box)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.setLayout(layout)

    def get_augmentation_count(self):
        return self.spin_box.value()



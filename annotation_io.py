import os
import json
import cv2
from enum import Enum
from typing import List, Dict, Tuple, Optional
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDialog, QVBoxLayout, QLabel, QSpinBox, QDialogButtonBox
import logging
import hashlib
import shutil
import yaml
import random
import numpy as np
import sys
import subprocess


class ImageAugmenter:
    def __init__(self):
        self.augmentation_types = [
            self._apply_horizontal_flip,
            self._apply_rotation,
            self._apply_zoom,
            self._apply_brightness,
            self._apply_translation,
            self._apply_noise
        ]
        self.augmentation_params = {
            'rotation': {'max_angle': 30},
            'zoom': {'range': (0.7, 1.5)},
            'brightness': {'intensity_range': (-50, 50)},
            'translation': {'max_shift': 0.1},
            'noise': {'intensity': 0.05}
        }

    def augment_image(self, img, boxes, n=5):
        """Generate n augmented versions of an image with valid bounding boxes"""
        results = []
        for i in range(n):
            img_copy = img.copy()
            boxes_copy = [list(box) for box in boxes]  # Convert to mutable lists

            # Apply 2 random transformations
            selected_transforms = random.sample(self.augmentation_types, 2)
            for transform in selected_transforms:
                img_copy, boxes_copy = transform(img_copy, boxes_copy)

            # Verify boxes are still valid
            valid_boxes = []
            h, w = img_copy.shape[:2]
            for box in boxes_copy:
                p1, p2, class_name = box
                if (0 <= p1.x() < w and 0 <= p1.y() < h and
                        0 <= p2.x() < w and 0 <= p2.y() < h and
                        abs(p1.x() - p2.x()) >= 5 and abs(p1.y() - p2.y()) >= 5):  # Minimum 5px size
                    valid_boxes.append((QPoint(p1.x(), p1.y()),
                                        QPoint(p2.x(), p2.y()),
                                        class_name))

            if valid_boxes:
                results.append((img_copy, valid_boxes))
        return results

    def _apply_horizontal_flip(self, img, boxes):
        h, w = img.shape[:2]
        img = cv2.flip(img, 1)
        for box in boxes:
            p1, p2, class_name = box
            new_x1 = w - p1.x()
            new_x2 = w - p2.x()
            box[0] = QPoint(min(new_x1, new_x2), p1.y())
            box[1] = QPoint(max(new_x1, new_x2), p2.y())
        return img, boxes

    def _apply_rotation(self, img, boxes, max_angle=None):
        angle = random.uniform(-self.augmentation_params['rotation']['max_angle'],
                               self.augmentation_params['rotation']['max_angle'])
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))
        for box in boxes:
            p1, p2, class_name = box
            corners = np.array([
                [p1.x(), p1.y()],
                [p1.x(), p2.y()],
                [p2.x(), p1.y()],
                [p2.x(), p2.y()]
            ], dtype=np.float32)
            rotated_corners = cv2.transform(np.array([corners]), M)[0]
            x_coords = rotated_corners[:, 0]
            y_coords = rotated_corners[:, 1]
            new_x1 = max(0, min(x_coords))
            new_y1 = max(0, min(y_coords))
            new_x2 = min(w, max(x_coords))
            new_y2 = min(h, max(y_coords))
            box[0] = QPoint(int(new_x1), int(new_y1))
            box[1] = QPoint(int(new_x2), int(new_y2))
        return img, boxes

    def _apply_zoom(self, img, boxes):
        zoom_range = self.augmentation_params['zoom']['range']
        zoom_factor = random.uniform(*zoom_range)
        h, w = img.shape[:2]
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        for box in boxes:
            p1, p2, class_name = box
            box[0] = QPoint(int(p1.x() * zoom_factor), int(p1.y() * zoom_factor))
            box[1] = QPoint(int(p2.x() * zoom_factor), int(p2.y() * zoom_factor))
        if zoom_factor > 1:
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            img = img[start_y:start_y + h, start_x:start_x + w]
            for box in boxes:
                p1, p2, class_name = box
                box[0] = QPoint(max(0, p1.x() - start_x), max(0, p1.y() - start_y))
                box[1] = QPoint(min(w, p2.x() - start_x), min(h, p2.y() - start_y))
        else:
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            img = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))
            for box in boxes:
                p1, p2, class_name = box
                box[0] = QPoint(p1.x() + pad_x, p1.y() + pad_y)
                box[1] = QPoint(p2.x() + pad_x, p2.y() + pad_y)
        return img, boxes

    def _apply_brightness(self, img, boxes):
        intensity = random.randint(*self.augmentation_params['brightness']['intensity_range'])
        img = np.clip(img.astype(np.int32) + intensity, 0, 255).astype(np.uint8)
        return img, boxes

    def _apply_translation(self, img, boxes):
        h, w = img.shape[:2]
        max_shift = self.augmentation_params['translation']['max_shift']
        max_x_shift = int(w * max_shift)
        max_y_shift = int(h * max_shift)
        shift_x = random.randint(-max_x_shift, max_x_shift)
        shift_y = random.randint(-max_y_shift, max_y_shift)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img = cv2.warpAffine(img, M, (w, h))
        for box in boxes:
            p1, p2, class_name = box
            box[0] = QPoint(p1.x() + shift_x, p1.y() + shift_y)
            box[1] = QPoint(p2.x() + shift_x, p2.y() + shift_y)
        return img, boxes

    def _apply_noise(self, img, boxes):
        intensity = self.augmentation_params['noise']['intensity']
        noise = np.random.normal(0, intensity * 255, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img, boxes


class DatasetExporter:
    def __init__(self):
        self.class_map = {}  # Maps class names to consistent IDs
        self.next_class_id = 0
        self.next_image_id = 1
        self.next_ann_id = 1

    def export_dataset(self, canvas, parent_window):
        """Main export function called from GUI"""
        export_root = QFileDialog.getExistingDirectory(
            parent_window, "Select Export Root Directory")
        if not export_root:
            return False

        try:
            # Create folder structure
            folders = self._create_export_structure(export_root)

            # Prepare COCO dataset structure
            coco_data = {
                "images": [],
                "annotations": [],
                "categories": [],
                "info": {"description": "Exported Dataset"},
                "licenses": [{"id": 1, "name": "Unknown"}]
            }

            # Process all annotated images
            success_count = 0
            class_names = set()  # Track unique class names

            for img_path, boxes in canvas.annotations.items():
                if not boxes:
                    continue

                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    # 1. Copy original image (no annotations)
                    self._copy_original_image(img_path, folders['images'])

                    # 2. Export annotations
                    self._export_coco(img_path, boxes, img.shape, folders['annotations'], coco_data)
                    self._export_yolo(img_path, boxes, img.shape, folders['yolo'])
                    self._export_custom(img_path, boxes, folders['custom'])
                    self._export_visual(img_path, boxes, folders['visual'], canvas.class_colors)

                    # Collect class names
                    for box in boxes:
                        class_names.add(box[2])  # box[2] is class_name

                    success_count += 1
                except Exception as e:
                    logging.error(f"Failed to export {img_path}: {str(e)}")

            # 3. Finalize exports
            # Save COCO dataset
            with open(os.path.join(folders['annotations'], 'instances_train.json'), 'w') as f:
                json.dump(coco_data, f, indent=2)

            # Generate YAML after all classes are collected
            self._generate_yaml(export_root, sorted(class_names))

            QMessageBox.information(parent_window, "Export Complete",
                                    f"Successfully exported {success_count} images")
            return True

        except Exception as e:
            QMessageBox.critical(parent_window, "Export Error",
                                 f"Dataset export failed: {str(e)}")
            return False

    def _generate_yaml(self, export_root, class_names):
        """Generate data.yaml for YOLO compatibility"""
        yaml_path = os.path.join(export_root, 'data.yaml')
        content = {
            'train': '../images/train',
            'val': '../images/val',  # Can be same as train if no val split
            'nc': len(class_names),
            'names': class_names
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(content, f, sort_keys=False)

    def _copy_original_image(self, img_path, target_folder):
        """Copy the original image to the target folder"""
        filename = os.path.basename(img_path)
        target_path = os.path.join(target_folder, filename)

        # Remove existing file if present
        if os.path.exists(target_path):
            os.remove(target_path)

        # Copy the file
        shutil.copy2(img_path, target_path)

    def _create_export_structure(self, export_root):
        """Create the required folder structure"""
        folders = {
            'root': export_root,
            'images': os.path.join(export_root, 'images', 'train'),
            'annotations': os.path.join(export_root, 'annotations'),
            'yolo': os.path.join(export_root, 'annotations', 'yolo'),
            'custom': os.path.join(export_root, 'annotations', 'custom_txt'),  # Changed from 'custom' to 'custom_txt'
            'coco': os.path.join(export_root, 'annotations', 'coco'),  # Added explicit coco folder
            'visual': os.path.join(export_root, 'visual_annotations')
        }

        for folder in folders.values():
            os.makedirs(folder, exist_ok=True)

        return folders

    def _export_coco(self, img_path, boxes, img_shape, ann_folder, coco_data):
        """Export to COCO format"""
        filename = os.path.basename(img_path)
        h, w, _ = img_shape

        # Check if image already exists
        existing_img = next((img for img in coco_data["images"] if img["file_name"] == filename), None)

        if existing_img:
            # Remove existing entries
            img_id = existing_img["id"]
            coco_data["images"] = [img for img in coco_data["images"] if img["id"] != img_id]
            coco_data["annotations"] = [ann for ann in coco_data["annotations"] if ann["image_id"] != img_id]
        else:
            img_id = self.next_image_id
            self.next_image_id += 1

        # Add image entry
        coco_data["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": w,
            "height": h,
            "license": 1,
            "date_captured": ""
        })

        # Add annotations
        for box in boxes:
            p1, p2, class_name = box
            x1, y1 = p1.x(), p1.y()
            width = abs(p2.x() - x1)
            height = abs(p2.y() - y1)

            # Get or create category
            cat_id = self._get_class_id(class_name)
            if not any(cat["id"] == cat_id for cat in coco_data["categories"]):
                coco_data["categories"].append({
                    "id": cat_id,
                    "name": class_name,
                    "supercategory": "object"
                })

            # Add annotation
            coco_data["annotations"].append({
                "id": self.next_ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
                "segmentation": []
            })
            self.next_ann_id += 1

    def _export_yolo(self, img_path, boxes, img_shape, yolo_folder):
        """Export to YOLO format"""
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(yolo_folder, f"{base_name}.txt")
        h, w = img_shape[:2]

        with open(output_path, 'w') as f:
            for box in boxes:
                p1, p2, class_name = box
                x1, y1 = p1.x(), p1.y()
                x2, y2 = p2.x(), p2.y()

                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = abs(x2 - x1) / w
                height = abs(y2 - y1) / h

                class_id = self._get_class_id(class_name)
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def _export_custom(self, img_path, boxes, custom_folder):
        """Export to custom TXT format"""
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(custom_folder, f"{base_name}.txt")

        with open(output_path, 'w') as f:
            for box in boxes:
                p1, p2, class_name = box
                x1, y1 = p1.x(), p1.y()
                width = abs(p2.x() - x1)
                height = abs(p2.y() - y1)
                f.write(f"{x1} {y1} {width} {height} {class_name}\n")

    def _export_visual(self, img_path, boxes, visual_folder, class_colors):
        """Export visual annotations"""
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(visual_folder, f"{base_name}_annotated.jpg")

        # Remove existing if present
        if os.path.exists(output_path):
            os.remove(output_path)

        img = cv2.imread(img_path)
        if img is None:
            return

        # Dynamic thickness based on image size
        thickness = max(1, int(0.002 * max(img.shape[:2])))

        for box in boxes:
            p1, p2, class_name = box
            x1, y1 = p1.x(), p1.y()
            x2, y2 = p2.x(), p2.y()

            color = class_colors.get(class_name, (0, 0, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            label = f"{class_name}"
            font_scale = 0.6
            text_thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
            cv2.rectangle(img, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), text_thickness, cv2.LINE_AA)

        cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    def _get_class_id(self, class_name):
        """Maintain consistent class IDs across exports"""
        if class_name not in self.class_map:
            self.class_map[class_name] = self.next_class_id
            self.next_class_id += 1
        return self.class_map[class_name]


def save_visual_annotation_image(
        img_path: str,
        boxes: List[Tuple[QPoint, QPoint, str]],
        output_folder: str,
        class_colors: Dict[str, Tuple[int, int, int]] = None
) -> bool:
    try:
        # Create output filename (ensure .jpg extension)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_filename = f"{base_name}_annotated.jpg"
        output_path = os.path.join(output_folder, output_filename)

        # Remove existing visual annotation if present
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError as e:
                logging.warning(f"Could not remove old visual annotation: {e}")

        # Load the original image
        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"Failed to read image for visual annotation: {img_path}")
            return False

        # Dynamic thickness based on image size
        thickness = max(1, int(0.002 * max(img.shape[:2])))

        # Draw each box on the image
        for box in boxes:
            p1, p2, class_name = box
            x1, y1 = p1.x(), p1.y()
            x2, y2 = p2.x(), p2.y()

            # Get color for this class (or default red)
            color = class_colors.get(class_name, (0, 0, 255)) if class_colors else (0, 0, 255)

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Draw label background
            label = f"{class_name}"
            font_scale = 0.6
            text_thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
            cv2.rectangle(img, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)

            # Draw label text
            cv2.putText(img, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), text_thickness, cv2.LINE_AA)

        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # Save with quality=95 to prevent recompression artifacts
        success = cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success:
            raise IOError(f"Failed to write image to {output_path}")
        return True

    except Exception as e:
        logging.error(f"Failed to save visual annotation for {img_path}: {str(e)}")
        return False


def _generate_color_for_class(class_name: str) -> Tuple[int, int, int]:
    """
    Generate a consistent RGB color for a given class name.
    The color is derived from the hash of the class name to ensure consistency.
    """
    # Hash the class name to get a deterministic value
    hash_digest = hashlib.md5(class_name.encode()).hexdigest()

    # Use parts of the hash to create RGB values
    r = int(hash_digest[0:2], 16)
    g = int(hash_digest[2:4], 16)
    b = int(hash_digest[4:6], 16)

    # Ensure the color is bright enough to be visible
    def adjust(c): return max(80, min(255, c + 60))

    return (adjust(r), adjust(g), adjust(b))


class AnnotationImporter:
    def __init__(self, canvas):
        self.canvas = canvas
        self.class_colors = canvas.class_colors if hasattr(canvas, 'class_colors') else {}

    def import_annotations(self, img_path: str) -> List[Tuple[QPoint, QPoint, str]]:
        """Main import function that tries all formats automatically"""
        base_path = os.path.splitext(img_path)[0]

        # Try YOLO format first
        yolo_path = f"{base_path}.txt"
        if os.path.exists(yolo_path):
            return self._import_yolo(yolo_path)

        # Try custom TXT format
        custom_path = f"{base_path}_custom.txt"
        if os.path.exists(custom_path):
            return self._import_custom_txt(custom_path)

        # Try COCO (requires separate handling)
        if hasattr(self.canvas, 'coco_data') and self.canvas.coco_data:
            return self._import_coco(img_path)

        return []

    def _import_yolo(self, file_path: str) -> List[Tuple[QPoint, QPoint, str]]:
        """Import YOLO format annotations (normalized coordinates)"""
        boxes = []
        try:
            with open(file_path, 'r') as f:
                img_width, img_height = self._get_image_dimensions(file_path)

                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height

                    # Convert to absolute coordinates
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    # Get class name (either from mapping or use ID)
                    class_name = self._get_class_name(class_id)

                    boxes.append((
                        QPoint(x1, y1),
                        QPoint(x2, y2),
                        class_name
                    ))
        except Exception as e:
            print(f"Error importing YOLO annotations: {str(e)}")
        return boxes

    def _import_coco(self, img_path: str) -> List[Tuple[QPoint, QPoint, str]]:
        """Import from COCO dataset format"""
        boxes = []
        try:
            img_name = os.path.basename(img_path)
            coco_data = self.canvas.coco_data

            # Find image in COCO data
            img_info = next((img for img in coco_data['images']
                             if img['file_name'] == img_name), None)
            if not img_info:
                return []

            img_id = img_info['id']
            img_width = img_info['width']
            img_height = img_info['height']

            # Get all annotations for this image
            for ann in coco_data['annotations']:
                if ann['image_id'] == img_id:
                    x, y, w, h = ann['bbox']
                    class_name = next(
                        (cat['name'] for cat in coco_data['categories']
                         if cat['id'] == ann['category_id']),
                        str(ann['category_id']))

                    boxes.append((
                        QPoint(int(x), int(y)),
                        QPoint(int(x + w), int(y + h)),
                        class_name
                    ))
        except Exception as e:
            print(f"Error importing COCO annotations: {str(e)}")
        return boxes

    def _import_custom_txt(self, file_path: str) -> List[Tuple[QPoint, QPoint, str]]:
        """Import custom TXT format (x1 y1 width height class)"""
        boxes = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    x1 = int(parts[0])
                    y1 = int(parts[1])
                    width = int(parts[2])
                    height = int(parts[3])
                    class_name = ' '.join(parts[4:])

                    boxes.append((
                        QPoint(x1, y1),
                        QPoint(x1 + width, y1 + height),
                        class_name
                    ))
        except Exception as e:
            print(f"Error importing custom annotations: {str(e)}")
        return boxes

    def _get_image_dimensions(self, annotation_path: str) -> Tuple[int, int]:
        """Get image dimensions from the corresponding image file"""
        img_path = os.path.splitext(annotation_path)[0] + '.jpg'  # Adjust extensions as needed
        if not os.path.exists(img_path):
            img_path = os.path.splitext(annotation_path)[0] + '.png'

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                return img.shape[1], img.shape[0]
        return 640, 480  # Fallback dimensions

    def _get_class_name(self, class_id: int) -> str:
        """Convert class ID to name, using canvas's class list if available"""
        if hasattr(self.canvas, 'class_list_widget'):
            item = self.canvas.class_list_widget.item(class_id)
            if item:
                return item.text()
        return f"class_{class_id}"


class AnnotationFormat(Enum):
    YOLO = "YOLO"
    COCO = "COCO"
    CUSTOM_TXT = "Custom TXT"


class AnnotationExporter:
    def __init__(self):
        self._class_mapping = {}  # {"class_name": class_id}
        self._next_class_id = 0

    def _get_class_id(self, class_name: str) -> int:
        """Maintain consistent class ID mapping across sessions"""
        if class_name not in self._class_mapping:
            self._class_mapping[class_name] = self._next_class_id
            self._next_class_id += 1
        return self._class_mapping[class_name]

    def _validate_boxes(self, boxes: List[Tuple], img_shape: Tuple[int, int]) -> None:
        """Validate bounding box coordinates and structure"""
        h, w, _ = img_shape
        for i, box in enumerate(boxes):
            if len(box) != 3:
                raise ValueError(f"Box {i} must be (QPoint, QPoint, label), got {len(box)} elements")

            p1, p2, label = box
            if not isinstance(p1, QPoint) or not isinstance(p2, QPoint):
                raise TypeError(f"Box {i} points must be QPoint objects")

            x1, y1 = p1.x(), p1.y()
            x2, y2 = p2.x(), p2.y()

            if not (0 <= x1 <= w and 0 <= x2 <= w):
                raise ValueError(f"Box {i} x-coordinates {x1}-{x2} out of image bounds (0-{w})")
            if not (0 <= y1 <= h and 0 <= y2 <= h):
                raise ValueError(f"Box {i} y-coordinates {y1}-{y2} out of image bounds (0-{h})")
            if abs(x1 - x2) < 5 or abs(y1 - y2) < 5:
                raise ValueError(f"Box {i} is too small (minimum 5px required)")

    def save_annotation_file(
            self,
            img_path: str,
            boxes: List[Tuple[QPoint, QPoint, str]],
            fmt: AnnotationFormat,
            folder: str,
            img_shape: Tuple[int, int, int],
            coco_data: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Save annotations in specified format.
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not boxes:
            logging.warning(f"No boxes to save for {img_path}")
            return coco_data

        self._validate_boxes(boxes, img_shape)
        os.makedirs(folder, exist_ok=True)

        try:
            if fmt == AnnotationFormat.YOLO:
                return self._save_yolo(img_path, boxes, folder, img_shape)
            elif fmt == AnnotationFormat.CUSTOM_TXT:
                return self._save_custom_txt(img_path, boxes, folder)
            elif fmt == AnnotationFormat.COCO:
                return self._save_coco(img_path, boxes, folder, img_shape, coco_data)
            else:
                raise ValueError(f"Unsupported format: {fmt}")
        except Exception as e:
            logging.error(f"Failed to save {fmt.value} annotations: {str(e)}")
            raise

    def _save_yolo(self, img_path, boxes, folder, img_shape):
        h, w, _ = img_shape
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(folder, f"{base_name}.txt")

        with open(output_path, 'w') as f:
            for box in boxes:
                p1, p2, label = box
                x1, y1 = p1.x(), p1.y()
                x2, y2 = p2.x(), p2.y()

                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = abs(x2 - x1) / w
                height = abs(y2 - y1) / h

                class_id = self._get_class_id(label)
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def _save_custom_txt(self, img_path, boxes, folder):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(folder, f"{base_name}.txt")

        with open(output_path, 'w') as f:
            for box in boxes:
                p1, p2, label = box
                x1, y1 = p1.x(), p1.y()
                width = abs(p2.x() - x1)
                height = abs(p2.y() - y1)
                f.write(f"{x1} {y1} {width} {height} {label}\n")

    def _save_coco(self, img_path, boxes, folder, img_shape, coco_data=None):
        h, w, _ = img_shape
        coco_data = coco_data or {
            "images": [],
            "annotations": [],
            "categories": [],
            "info": {"description": "COCO dataset"},
            "licenses": [{"id": 1, "name": "Unknown"}]
        }

        image_id = len(coco_data["images"]) + 1
        coco_data["images"].append({
            "id": image_id,
            "file_name": os.path.basename(img_path),
            "width": w,
            "height": h,
            "license": 1,
            "date_captured": ""
        })

        for box in boxes:
            p1, p2, label = box
            x1, y1 = p1.x(), p1.y()
            width = abs(p2.x() - x1)
            height = abs(p2.y() - y1)

            cat_id = next((cat["id"] for cat in coco_data["categories"] if cat["name"] == label), None)
            if cat_id is None:
                cat_id = len(coco_data["categories"]) + 1
                coco_data["categories"].append({
                    "id": cat_id,
                    "name": label,
                    "supercategory": "object"
                })

            coco_data["annotations"].append({
                "id": len(coco_data["annotations"]) + 1,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
                "segmentation": []
            })

        # Save COCO annotations
        output_path = os.path.join(folder, "annotations.json")
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        return coco_data


# Singleton exporter instance
exporter = AnnotationExporter()


def save_annotation_file(
        img_path: str,
        boxes: List[Tuple[QPoint, QPoint, str]],
        fmt: AnnotationFormat,
        folder: str,
        img_shape: Tuple[int, int, int],
        coco_data: Optional[Dict] = None
) -> Optional[Dict]:
    return exporter.save_annotation_file(
        img_path, boxes, fmt, folder, img_shape, coco_data
    )

import os
import json
from enum import Enum
from typing import List, Dict, Tuple, Optional
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QFileDialog
import logging
import hashlib


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
        output_path = os.path.join(folder, f"{base_name}_custom.txt")

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

        # Ask user for filename before saving
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save COCO Annotations As",
            os.path.join(folder, "annotations.json"),
            "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return coco_data  # User canceled

        with open(file_path, 'w') as f:
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



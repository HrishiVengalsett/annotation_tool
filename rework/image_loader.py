import os
import logging
from typing import List, Optional

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")


def load_image_folder(folder_path: str) -> List[str]:
    """Return a sorted list of image paths in the given folder."""
    if not os.path.isdir(folder_path):
        logging.error(f"Folder does not exist: {folder_path}")
        return []

    images = []
    for f in os.listdir(folder_path):
        ext = os.path.splitext(f)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            images.append(os.path.join(folder_path, f))

    if not images:
        logging.warning(f"No supported images found in {folder_path}")

    return sorted(images)


def get_image_at_index(image_paths: List[str], index: int) -> Optional[str]:
    """Return image path at a specific index, or None if out of range."""
    if 0 <= index < len(image_paths):
        return image_paths[index]
    return None


def get_next_index(current_index: int, total_images: int) -> int:
    """Return the next image index, or current if at end."""
    return min(current_index + 1, total_images - 1)


def get_prev_index(current_index: int) -> int:
    """Return the previous image index, or current if at start."""
    return max(current_index - 1, 0)


def add_image_to_list(image_list: List[str], image_path: str) -> List[str]:
    """Add image to list if not already present."""
    if image_path and image_path not in image_list:
        image_list.append(image_path)
    return image_list


def load_image_folders(folder_paths: List[str]) -> List[str]:
    """Load images from multiple folders."""
    return [img for folder in folder_paths for img in load_image_folder(folder)]
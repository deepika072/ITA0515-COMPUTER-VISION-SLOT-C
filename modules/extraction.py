from io import BytesIO

import cv2
import numpy as np
from PIL import Image


MAX_EXTRACTION_PIXELS = 20_000_000
MAX_SEGMENTATION_SIDE = 960


def extract_foreground(image_bytes, bounding_box, iterations=5):
    """Return a transparent PNG and its binary foreground mask."""
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            rgb_image = np.asarray(image.convert("RGB"))
    except (OSError, ValueError) as exc:
        raise ValueError("The uploaded image could not be opened for extraction.") from exc

    image_height, image_width = rgb_image.shape[:2]
    if image_width * image_height > MAX_EXTRACTION_PIXELS:
        raise ValueError(
            "This image is too large to segment safely. Upload an image with "
            "20 megapixels or fewer."
        )

    x, y, width, height = bounding_box
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > image_width
        or y + height > image_height
    ):
        raise ValueError("The selection rectangle must fit inside the image.")
    if width == image_width and height == image_height:
        raise ValueError("Leave some background outside the selection rectangle.")

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    scale = min(1.0, MAX_SEGMENTATION_SIDE / max(image_width, image_height))
    if scale < 1.0:
        scaled_width = max(1, round(image_width * scale))
        scaled_height = max(1, round(image_height * scale))
        bgr_image = cv2.resize(
            bgr_image,
            (scaled_width, scaled_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        scaled_width, scaled_height = image_width, image_height

    scaled_x = round(x * scaled_width / image_width)
    scaled_y = round(y * scaled_height / image_height)
    scaled_right = round((x + width) * scaled_width / image_width)
    scaled_bottom = round((y + height) * scaled_height / image_height)
    scaled_rectangle = (
        scaled_x,
        scaled_y,
        max(1, scaled_right - scaled_x),
        max(1, scaled_bottom - scaled_y),
    )
    grabcut_mask = np.zeros((scaled_height, scaled_width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            bgr_image,
            grabcut_mask,
            scaled_rectangle,
            background_model,
            foreground_model,
            iterations,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error as exc:
        raise ValueError(
            "The foreground could not be segmented. Try a tighter rectangle "
            "that leaves some background visible."
        ) from exc

    foreground_mask = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    if foreground_mask.shape != (image_height, image_width):
        foreground_mask = cv2.resize(
            foreground_mask,
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        )
    rgba_image = np.dstack((rgb_image, foreground_mask))

    output = BytesIO()
    Image.fromarray(rgba_image).save(output, format="PNG")
    return output.getvalue(), foreground_mask
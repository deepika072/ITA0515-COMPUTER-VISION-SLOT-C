from io import BytesIO
import warnings

from PIL import Image, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
}


def validate_image(filename, image_bytes):
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    expected_format = SUPPORTED_EXTENSIONS.get(extension)
    if expected_format is None:
        raise ValueError("Choose a JPG, JPEG, or PNG image.")
    if not image_bytes:
        raise ValueError("The selected file is empty.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as image:
                image_format = image.format
                width, height = image.size
                image.verify()
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        SyntaxError,
        UnidentifiedImageError,
    ) as exc:
        raise ValueError(
            "The selected file is not a valid image or is too large to safely process."
        ) from exc

    if image_format != expected_format:
        raise ValueError("The file extension does not match the image content.")

    return {
        "filename": filename,
        "format": image_format,
        "width": width,
        "height": height,
        "size_bytes": len(image_bytes),
    }


def format_file_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"

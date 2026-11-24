from io import BytesIO

from PIL import Image, ImageColor


BACKGROUND_MODES = ("Transparent", "White", "Custom color")


def apply_background(extracted_image, mode, color="#ffffff"):
    """Composite a transparent cutout on the selected background."""
    if mode not in BACKGROUND_MODES:
        raise ValueError("Choose a supported background style.")

    try:
        with Image.open(BytesIO(extracted_image)) as image:
            foreground = image.convert("RGBA")
    except (OSError, ValueError) as exc:
        raise ValueError("The extracted image could not be opened.") from exc

    if mode == "Transparent":
        result = foreground
    else:
        if mode == "White":
            color = "#ffffff"
        try:
            rgb_color = ImageColor.getrgb(color)
        except ValueError as exc:
            raise ValueError("Choose a valid background color.") from exc

        background = Image.new("RGBA", foreground.size, (*rgb_color, 255))
        result = Image.alpha_composite(background, foreground).convert("RGB")

    output = BytesIO()
    result.save(output, format="PNG")
    return output.getvalue()
from datetime import datetime


MAX_HISTORY_ITEMS = 10
MAX_HISTORY_BYTES = 50 * 1024 * 1024


def _entry_size(entry):
    return len(entry["extracted_image"]) + len(entry.get("edited_image") or b"")


def _trim_history(history):
    while len(history) > MAX_HISTORY_ITEMS or sum(
        _entry_size(entry) for entry in history
    ) > MAX_HISTORY_BYTES:
        history.pop()


def record_extraction(
    history,
    identifier,
    image_filename,
    object_name,
    extracted_image,
):
    if len(extracted_image) > MAX_HISTORY_BYTES:
        return False

    entry = {
        "id": identifier,
        "image_filename": image_filename,
        "object_name": object_name or "Object",
        "created_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "extracted_image": extracted_image,
        "edited_image": None,
        "background_mode": None,
    }
    history[:] = [item for item in history if item["id"] != identifier]
    history.insert(0, entry)
    _trim_history(history)
    return any(item["id"] == identifier for item in history)


def record_background(history, identifier, edited_image, background_mode):
    entry = next((item for item in history if item["id"] == identifier), None)
    if entry is None or (
        len(entry["extracted_image"]) + len(edited_image) > MAX_HISTORY_BYTES
    ):
        return False

    entry["edited_image"] = edited_image
    entry["background_mode"] = background_mode
    _trim_history(history)
    return True

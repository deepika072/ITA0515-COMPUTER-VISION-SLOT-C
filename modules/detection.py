from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import re
import shutil
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import onnxruntime as ort
from PIL import Image


MODEL_URL = (
    "https://huggingface.co/onnxmodelzoo/ssd_mobilenet_v1_12-int8/resolve/"
    "929618539097dbeb779c13aed75dfe346d016d48/ssd_mobilenet_v1_12-int8.onnx"
)
MODEL_FILENAME = "ssd_mobilenet_v1_12-int8.onnx"
MODEL_SIZE_BYTES = 9_540_809
MODEL_SHA256 = "2b79e6a7fb1ec6a33f332b9b10d82d9de4b7b49dcd26b5946921bb356895c954"
MAX_DETECTION_PIXELS = 20_000_000
MAX_DETECTION_SIDE = 640
DEFAULT_CONFIDENCE = 0.25
_ONNX_RUNTIME_ERRORS = (
    ort.capi.onnxruntime_pybind11_state.Fail,
    ort.capi.onnxruntime_pybind11_state.InvalidArgument,
    ort.capi.onnxruntime_pybind11_state.InvalidGraph,
    ort.capi.onnxruntime_pybind11_state.InvalidProtobuf,
    ort.capi.onnxruntime_pybind11_state.NoModel,
    ort.capi.onnxruntime_pybind11_state.NoSuchFile,
    ort.capi.onnxruntime_pybind11_state.NotFound,
    ort.capi.onnxruntime_pybind11_state.RuntimeException,
)

COCO_CLASS_NAMES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

QUERY_ALIASES = {
    "aeroplane": "airplane",
    "bike": "bicycle",
    "cellphone": "cell phone",
    "fridge": "refrigerator",
    "hair dryer": "hair drier",
    "hairdryer": "hair drier",
    "kitten": "cat",
    "mobile phone": "cell phone",
    "motorbike": "motorcycle",
    "phone": "cell phone",
    "plant": "potted plant",
    "puppy": "dog",
    "sofa": "couch",
    "television": "tv",
    "teddy": "teddy bear",
}

QUERY_STOP_WORDS = {
    "a",
    "an",
    "find",
    "i",
    "me",
    "object",
    "only",
    "please",
    "the",
    "to",
    "want",
}


def _model_path():
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        cache_root = Path(local_app_data) / "SmartVisionAI"
    else:
        cache_root = Path.home() / ".cache" / "smart-vision-ai"
    return cache_root / "models" / MODEL_FILENAME


def _file_sha256(path):
    digest = sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_model():
    model_path = _model_path()
    if model_path.is_file():
        if (
            model_path.stat().st_size == MODEL_SIZE_BYTES
            and _file_sha256(model_path) == MODEL_SHA256
        ):
            return model_path
        model_path.unlink()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = model_path.with_suffix(model_path.suffix + ".part")
    request = Request(MODEL_URL, headers={"User-Agent": "SmartVisionAI/1.0"})
    try:
        with urlopen(request, timeout=120) as response:
            with partial_path.open("wb") as downloaded_model:
                shutil.copyfileobj(response, downloaded_model)
    except (OSError, URLError) as exc:
        if partial_path.exists():
            partial_path.unlink()
        raise RuntimeError(
            "The local object detector could not be downloaded. Check your "
            "internet connection and retry."
        ) from exc

    if (
        partial_path.stat().st_size != MODEL_SIZE_BYTES
        or _file_sha256(partial_path) != MODEL_SHA256
    ):
        partial_path.unlink()
        raise RuntimeError(
            "The downloaded detector did not pass its integrity check. Retry "
            "the model download."
        )

    os.replace(partial_path, model_path)
    return model_path


@lru_cache(maxsize=1)
def _load_detector():
    model_path = _ensure_model()
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    output_names = {output.name for output in session.get_outputs()}
    required_outputs = {
        "detection_boxes",
        "detection_classes",
        "detection_scores",
        "num_detections",
    }
    if not required_outputs.issubset(output_names):
        raise RuntimeError("The downloaded object detector has an unexpected format.")
    return session


def detect_objects(image_bytes, min_confidence=DEFAULT_CONFIDENCE):
    """Detect common COCO objects and return labels, scores, and pixel boxes."""
    if not 0 <= min_confidence <= 1:
        raise ValueError("The confidence threshold must be between 0 and 1.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            source_image = image.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError("The uploaded image could not be opened for detection.") from exc

    image_width, image_height = source_image.size
    if image_width * image_height > MAX_DETECTION_PIXELS:
        raise ValueError(
            "This image is too large to analyze safely. Upload an image with "
            "20 megapixels or fewer."
        )

    inference_image = source_image.copy()
    inference_image.thumbnail(
        (MAX_DETECTION_SIDE, MAX_DETECTION_SIDE),
        Image.Resampling.BILINEAR,
    )
    image_tensor = np.expand_dims(np.asarray(inference_image, dtype=np.uint8), axis=0)

    try:
        session = _load_detector()
        input_name = session.get_inputs()[0].name
        output_names = (
            "detection_boxes",
            "detection_classes",
            "detection_scores",
            "num_detections",
        )
        outputs = session.run(output_names, {input_name: image_tensor})
    except _ONNX_RUNTIME_ERRORS as exc:
        raise RuntimeError(
            "The local object detector failed to load or analyze this image. "
            "Retry the search or continue with a manual rectangle."
        ) from exc
    output_by_name = dict(zip(output_names, outputs))
    boxes = output_by_name["detection_boxes"][0]
    classes = output_by_name["detection_classes"][0]
    scores = output_by_name["detection_scores"][0]
    detection_count = min(
        int(output_by_name["num_detections"][0]),
        len(boxes),
        len(classes),
        len(scores),
    )

    detections = []
    for index in range(detection_count):
        score = float(scores[index])
        class_id = int(classes[index])
        label = COCO_CLASS_NAMES.get(class_id)
        if label is None or score < min_confidence:
            continue

        top, left, bottom, right = (float(value) for value in boxes[index])
        x1 = max(0, min(image_width, round(left * image_width)))
        y1 = max(0, min(image_height, round(top * image_height)))
        x2 = max(0, min(image_width, round(right * image_width)))
        y2 = max(0, min(image_height, round(bottom * image_height)))
        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            {
                "label": label,
                "score": score,
                "box": (x1, y1, x2, y2),
            }
        )

    return sorted(detections, key=lambda detection: detection["score"], reverse=True)


def _normalize_query(query):
    words = re.findall(r"[a-z0-9]+", query.casefold())
    words = [word for word in words if word not in QUERY_STOP_WORDS]
    normalized = " ".join(words)
    normalized = QUERY_ALIASES.get(normalized, normalized)
    return set(normalized.split())


def find_matching_objects(image_bytes, query, min_confidence=DEFAULT_CONFIDENCE):
    """Return detections whose COCO labels match the user's object name."""
    query_words = _normalize_query(query)
    if not query_words:
        raise ValueError("Enter an object name to search for.")

    matches = [
        detection
        for detection in detect_objects(image_bytes, min_confidence)
        if set(detection["label"].split()).issubset(query_words)
    ]
    return matches
from hashlib import sha1
from io import BytesIO
import re

import streamlit as st
from PIL import Image, ImageDraw

from modules.extraction import extract_foreground
from modules.history import record_extraction
from modules.ui import initialize_session_state, render_page_header, render_sidebar


st.set_page_config(
    page_title="Object Extraction | Smart Vision AI",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_session_state()
render_sidebar()
render_page_header(
    "Object extraction",
    "Refine an object selection and export a transparent cutout.",
)

if st.session_state["uploaded_image"] is None:
    st.warning("Upload an image before extracting an object.")
    if st.button("Go to image upload", type="primary"):
        st.switch_page("pages/2_Upload_Image.py")
else:
    image_bytes = st.session_state["uploaded_image"]
    object_name = st.session_state["extraction_label"].strip()
    selected_object = st.session_state.get("selected_object")
    with Image.open(BytesIO(image_bytes)) as image:
        source_image = image.convert("RGB")
    image_width, image_height = source_image.size
    image_id = sha1(image_bytes).hexdigest()[:10]
    if selected_object is not None:
        x1, y1, x2, y2 = selected_object["box"]
        initial_x, initial_y = x1, y1
        initial_width, initial_height = max(1, x2 - x1), max(1, y2 - y1)
    else:
        initial_x, initial_y = image_width // 10, image_height // 10
        initial_width = max(1, image_width * 4 // 5)
        initial_height = max(1, image_height * 4 // 5)
    selection_id = sha1(
        f"{image_id}:{initial_x},{initial_y},{initial_width},{initial_height}".encode(
            "ascii"
        )
    ).hexdigest()[:10]

    st.markdown(
        "Review the detected object's rectangle and adjust it if needed. "
        "Keep some background visible outside it so GrabCut can refine the edges."
    )
    if selected_object is not None:
        st.caption(
            f"Target: {object_name}. Detected as {selected_object['label']} "
            f"({selected_object['score']:.0%} confidence)."
        )
    else:
        st.caption(
            f"Target: {object_name or 'unnamed object'}. No matching detection "
            "was selected; refine the rectangle manually."
        )

    controls, preview = st.columns([1, 1.5], gap="large")
    with controls:
        st.subheader("Selection rectangle")
        x = st.number_input(
            "Left (px)",
            min_value=0,
            max_value=image_width - 1,
            value=initial_x,
            key=f"extraction_x_{selection_id}",
        )
        y = st.number_input(
            "Top (px)",
            min_value=0,
            max_value=image_height - 1,
            value=initial_y,
            key=f"extraction_y_{selection_id}",
        )
        width = st.number_input(
            "Width (px)",
            min_value=1,
            max_value=image_width,
            value=initial_width,
            key=f"extraction_width_{selection_id}",
        )
        height = st.number_input(
            "Height (px)",
            min_value=1,
            max_value=image_height,
            value=initial_height,
            key=f"extraction_height_{selection_id}",
        )

    x, y, width, height = map(int, (x, y, width, height))
    with preview:
        st.subheader("Original image")
        marked_image = source_image.copy()
        ImageDraw.Draw(marked_image).rectangle(
            (x, y, x + width - 1, y + height - 1),
            outline="#a78bfa",
            width=max(2, min(image_width, image_height) // 250),
        )
        st.image(marked_image, caption="Adjust the rectangle to enclose the object.")

    extraction_signature = None
    if x + width > image_width or y + height > image_height:
        st.error("The selection rectangle extends beyond the image. Adjust its position or size.")
    else:
        extraction_signature = sha1(
            image_bytes + f"{x},{y},{width},{height}".encode("ascii")
        ).hexdigest()
        if st.session_state.get("extraction_signature") != extraction_signature:
            st.session_state["extracted_image"] = None
            st.session_state["masks"] = None
            st.session_state["extraction_signature"] = extraction_signature

        if st.button("Extract object", type="primary"):
            with st.spinner("Segmenting the selected region..."):
                try:
                    extracted_image, foreground_mask = extract_foreground(
                        image_bytes,
                        (x, y, width, height),
                        iterations=2,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["extracted_image"] = extracted_image
                    st.session_state["masks"] = foreground_mask
                    if extraction_signature not in st.session_state[
                        "processed_extraction_ids"
                    ]:
                        st.session_state["processed_extraction_ids"].append(
                            extraction_signature
                        )
                        st.session_state["objects_extracted"] = len(
                            st.session_state["processed_extraction_ids"]
                        )
                    image_filename = st.session_state["image_info"]["filename"]
                    history_saved = record_extraction(
                        st.session_state["processing_history"],
                        extraction_signature,
                        image_filename,
                        object_name,
                        extracted_image,
                    )
                    st.success("Object extracted. Review the transparent cutout below.")
                    if not history_saved:
                        st.warning(
                            "This cutout exceeds the session history storage limit. "
                            "Download it before leaving this page."
                        )

    extracted_image = st.session_state["extracted_image"]
    if (
        extracted_image is not None
        and extraction_signature is not None
        and st.session_state.get("extraction_signature") == extraction_signature
    ):
        result, download = st.columns([1, 1])
        with result:
            st.subheader("Transparent cutout")
            st.image(extracted_image, caption="Transparent PNG preview")
        with download:
            st.subheader("Export")
            st.download_button(
                "Download transparent PNG",
                data=extracted_image,
                file_name=(
                    re.sub(r"[^A-Za-z0-9._-]+", "-", object_name).strip("-.")[:60]
                    or "extracted-object"
                )
                + ".png",
                mime="image/png",
                type="primary",
            )
            st.page_link("pages/6_Background_Editor.py", label="Continue to background editor")
import streamlit as st

from modules.image_validation import format_file_size, validate_image
from modules.ui import initialize_session_state, render_page_header, render_sidebar


st.set_page_config(
    page_title="Upload Image | Smart Vision AI",
    page_icon="📤",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_session_state()
render_sidebar()
render_page_header(
    "Upload an image",
    "Start with a clear photo. You will be able to review its details before analysis.",
)

st.markdown("Choose a JPG, JPEG, or PNG file. The image is validated before it enters the workflow.")
uploaded_file = st.file_uploader(
    "Drop an image here or browse files",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, and PNG.",
)

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    try:
        image_info = validate_image(uploaded_file.name, image_bytes)
    except ValueError as exc:
        st.error(str(exc))
    else:
        existing_info = st.session_state["image_info"]
        is_new_image = (
            image_bytes != st.session_state["uploaded_image"]
            or existing_info is None
            or existing_info["filename"] != image_info["filename"]
        )
        if is_new_image:
            st.session_state["images_processed"] += 1
            st.session_state["uploaded_image"] = image_bytes
            st.session_state["image_info"] = image_info
            st.session_state["detections"] = []
            st.session_state["matching_detections"] = []
            st.session_state["detection_query"] = ""
            st.session_state["detection_signature"] = None
            st.session_state["detection_error_signature"] = None
            st.session_state["detection_error_message"] = ""
            st.session_state["prompt"] = ""
            st.session_state["extraction_label"] = ""
            st.session_state["selected_objects"] = []
            st.session_state["selected_object"] = None
            st.session_state["masks"] = None
            st.session_state["extracted_image"] = None
            st.session_state["edited_image"] = None
            st.success("Image validated. Opening analysis…")
            st.switch_page("pages/3_Analysis.py")
        else:
            st.success("This image is already loaded in your workspace.")
            st.image(image_bytes, caption=image_info["filename"])
            st.caption(
                f'{image_info["width"]} × {image_info["height"]} px · '
                f'{image_info["format"]} · {format_file_size(image_info["size_bytes"])}'
            )

if st.session_state["uploaded_image"] is not None:
    if st.button("Continue to analysis", type="primary"):
        st.switch_page("pages/3_Analysis.py")
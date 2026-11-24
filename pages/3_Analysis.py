import streamlit as st

from modules.image_validation import format_file_size

from modules.ui import initialize_session_state, render_page_header, render_sidebar


st.set_page_config(
    page_title="Analysis | Smart Vision AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_session_state()
render_sidebar()
render_page_header(
    "Image analysis",
    "Review image details before searching for objects to extract.",
)

if st.session_state["uploaded_image"] is None:
    st.warning("Upload an image before opening the analysis workspace.")
    if st.button("Go to image upload", type="primary"):
        st.switch_page("pages/2_Upload_Image.py")
else:
    image_info = st.session_state["image_info"]
    st.info(
        "Your image is ready. Search for common objects by name with the local "
        "detector on the next step."
    )
    preview, details = st.columns([1.5, 1], gap="large")
    with preview:
        st.image(
            st.session_state["uploaded_image"],
            caption=image_info["filename"],
        )
    with details:
        st.subheader("Image details")
        st.metric("Dimensions", f'{image_info["width"]} × {image_info["height"]} px')
        st.metric("Format", image_info["format"])
        st.metric("File size", format_file_size(image_info["size_bytes"]))
        st.caption(f'File: {image_info["filename"]}')
        st.subheader("Detection")
        st.caption("The detector runs locally and can identify common COCO objects.")
    if st.button("Continue to object selection", type="primary"):
        st.switch_page("pages/4_Object_Selection.py")
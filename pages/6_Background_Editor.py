from hashlib import sha1

import streamlit as st

from modules.background import BACKGROUND_MODES, apply_background
from modules.history import record_background
from modules.ui import initialize_session_state, render_page_header, render_sidebar


st.set_page_config(
    page_title="Background Editor | Smart Vision AI",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_session_state()
render_sidebar()
render_page_header(
    "Background editor",
    "Choose a clean transparent, white, or custom background for your extracted object.",
)

if st.session_state["extracted_image"] is None:
    st.warning("Extract an object first to use the background editor.")
    if st.button("Go to object extraction", type="primary"):
        st.switch_page("pages/5_Object_Extraction.py")
else:
    extracted_image = st.session_state["extracted_image"]
    st.markdown(
        "Preview your cutout on a transparent, white, or custom solid-color "
        "background, then download the finished PNG."
    )

    settings, preview = st.columns([1, 1.5], gap="large")
    with settings:
        st.subheader("Background")
        background_mode = st.selectbox("Style", BACKGROUND_MODES)
        background_color = "#ffffff"
        if background_mode == "Custom color":
            background_color = st.color_picker(
                "Choose a background color",
                value="#dbeafe",
            )

    signature_value = f"{background_mode}:{background_color}"
    background_signature = sha1(
        extracted_image + signature_value.encode("utf-8")
    ).hexdigest()
    if st.session_state.get("background_signature") != background_signature:
        st.session_state["edited_image"] = None
        st.session_state["background_signature"] = background_signature

    with preview:
        st.subheader("Extracted object")
        st.image(extracted_image, caption="Transparent cutout")

    if st.button("Apply background", type="primary"):
        try:
            edited_image = apply_background(
                extracted_image,
                background_mode,
                background_color,
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state["edited_image"] = edited_image
            history_saved = record_background(
                st.session_state["processing_history"],
                st.session_state.get("extraction_signature"),
                edited_image,
                background_mode,
            )
            st.success("Background applied. Your PNG is ready to download.")
            if not history_saved:
                st.warning(
                    "The final image could not be added to session history. "
                    "Download it before leaving this page."
                )

    edited_image = st.session_state["edited_image"]
    if (
        edited_image is not None
        and st.session_state.get("background_signature") == background_signature
    ):
        result, download = st.columns([1, 1], gap="large")
        with result:
            st.subheader("Finished image")
            st.image(edited_image, caption=f"{background_mode} background")
        with download:
            st.subheader("Export")
            st.download_button(
                "Download PNG",
                data=edited_image,
                file_name="smart-vision-result.png",
                mime="image/png",
                type="primary",
            )
            st.page_link("pages/5_Object_Extraction.py", label="Adjust the selection")
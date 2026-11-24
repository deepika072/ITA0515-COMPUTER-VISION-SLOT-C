import streamlit as st

from modules.ui import initialize_session_state, render_page_header, render_sidebar


st.set_page_config(
    page_title="History | Smart Vision AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_session_state()
render_sidebar()
render_page_header(
    "Processing history",
    "Review and download recent cutouts and finished compositions.",
)

history = st.session_state["processing_history"]
st.caption(
    "History is stored in this app session only. It keeps up to 10 results "
    "within a 50 MB limit and clears when the session ends."
)

if not history:
    st.info("No extractions in this session yet.")
else:
    if st.button("Clear history"):
        history.clear()
        st.rerun()

    for entry in history:
        with st.container(border=True):
            details, preview = st.columns([1, 1.2], gap="large")
            with details:
                st.subheader(entry["object_name"])
                st.caption(
                    f'{entry["image_filename"]} · {entry["created_at"]}'
                )
                if entry["background_mode"]:
                    st.caption(f'{entry["background_mode"]} background applied')
                st.download_button(
                    "Download transparent cutout",
                    data=entry["extracted_image"],
                    file_name=f'{entry["id"][:12]}-cutout.png',
                    mime="image/png",
                    key=f'history_cutout_{entry["id"]}',
                )
                if entry["edited_image"] is not None:
                    st.download_button(
                        "Download finished image",
                        data=entry["edited_image"],
                        file_name=f'{entry["id"][:12]}-result.png',
                        mime="image/png",
                        key=f'history_result_{entry["id"]}',
                    )
                if st.button(
                    "Open cutout in background editor",
                    key=f'history_open_{entry["id"]}',
                ):
                    st.session_state["extracted_image"] = entry["extracted_image"]
                    st.session_state["edited_image"] = None
                    st.session_state["extraction_label"] = entry["object_name"]
                    st.session_state["extraction_signature"] = entry["id"]
                    st.session_state["background_signature"] = None
                    st.switch_page("pages/6_Background_Editor.py")
            with preview:
                preview_image = entry["edited_image"] or entry["extracted_image"]
                preview_caption = (
                    "Finished composition"
                    if entry["edited_image"] is not None
                    else "Transparent cutout"
                )
                st.image(preview_image, caption=preview_caption)
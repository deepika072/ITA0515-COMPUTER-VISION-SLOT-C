
from hashlib import sha1

import streamlit as st

from modules.detection import COCO_CLASS_NAMES, find_matching_objects
from modules.ui import initialize_session_state, render_page_header, render_sidebar


st.set_page_config(
    page_title="Object Selection | Smart Vision AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_session_state()
render_sidebar()
render_page_header(
    "Choose what to extract",
    "Search for a common object by name, then choose which detected instance to extract.",
)

category_names = list(COCO_CLASS_NAMES.values())
with st.expander(f"See all {len(category_names)} supported object categories"):
    columns = st.columns(4)
    categories_per_column = (len(category_names) + len(columns) - 1) // len(columns)
    for index, name in enumerate(category_names):
        display_name = "TV" if name == "tv" else name.title()
        columns[index // categories_per_column].markdown(f"- {display_name}")

if st.session_state["uploaded_image"] is None:
    st.warning("Upload an image before choosing an object.")
    if st.button("Go to image upload", type="primary"):
        st.switch_page("pages/2_Upload_Image.py")
else:
    image_info = st.session_state["image_info"]
    image_bytes = st.session_state["uploaded_image"]
    preview, selection = st.columns([1.5, 1], gap="large")
    with preview:
        st.image(image_bytes, caption=image_info["filename"])
    with selection:
        st.subheader("What do you want to extract?")
        with st.form("object_search_form"):
            object_name = st.text_input(
                "Object name",
                key="prompt",
                placeholder="For example: dog",
            )
            search_submitted = st.form_submit_button(
                "Find matching objects",
                type="primary",
            )

        st.caption(
            f"Search runs locally across {len(COCO_CLASS_NAMES)} common object "
            "categories. The first "
            "search downloads and initializes the detector (about 9 MB); later "
            "searches are faster."
        )
        st.markdown(
            "[Detector model and Apache-2.0 license]"
            "(https://huggingface.co/onnxmodelzoo/ssd_mobilenet_v1_12-int8)"
        )

        if search_submitted:
            if not object_name.strip():
                st.error("Enter an object name to continue.")
            else:
                query = object_name.strip()
                query_signature = sha1(
                    image_bytes + query.casefold().encode("utf-8")
                ).hexdigest()
                st.session_state["extraction_label"] = query
                st.session_state["detection_signature"] = None
                st.session_state["detection_error_signature"] = None
                st.session_state["detection_error_message"] = ""
                st.session_state["detections"] = []
                st.session_state["matching_detections"] = []
                st.session_state["selected_objects"] = []
                st.session_state["selected_object"] = None
                try:
                    with st.spinner(
                        "Loading the local detector and searching for matches..."
                    ):
                        matches = find_matching_objects(image_bytes, query)
                except (OSError, RuntimeError, ValueError) as exc:
                    st.session_state["detection_error_signature"] = query_signature
                    st.session_state["detection_error_message"] = str(exc)
                else:
                    st.session_state["detections"] = matches
                    st.session_state["matching_detections"] = matches
                    st.session_state["detection_query"] = query
                    st.session_state["detection_signature"] = query_signature
                    st.session_state["selected_objects"] = []
                    st.session_state["selected_object"] = None

        query_signature = sha1(
            image_bytes + object_name.strip().casefold().encode("utf-8")
        ).hexdigest()
        if st.session_state.get("detection_error_signature") == query_signature:
            st.error(st.session_state["detection_error_message"])
            if st.button("Continue with manual rectangle"):
                st.session_state["selected_object"] = None
                st.session_state["selected_objects"] = []
                st.switch_page("pages/5_Object_Extraction.py")
        elif (
            object_name.strip()
            and st.session_state.get("detection_signature") == query_signature
        ):
            matches = st.session_state["matching_detections"]
            if matches:
                st.success(f"Found {len(matches)} matching object(s).")
                selected_index = st.selectbox(
                    "Choose an instance",
                    options=range(len(matches)),
                    format_func=lambda index: (
                        f'{matches[index]["label"].title()} · '
                        f'{matches[index]["score"]:.0%} confidence · '
                        f'box {matches[index]["box"]}'
                    ),
                    key=f"matching_object_{query_signature}",
                )
                if st.button("Extract selected object", type="primary"):
                    selected_object = matches[selected_index]
                    st.session_state["selected_object"] = selected_object
                    st.session_state["selected_objects"] = [selected_object]
                    st.session_state["extraction_label"] = object_name.strip()
                    st.switch_page("pages/5_Object_Extraction.py")
            else:
                st.warning(
                    f"No {object_name.strip()!r} matches were detected. "
                    "Try a common object name or continue with a manual rectangle."
                )
                if st.button("Continue with manual rectangle"):
                    st.session_state["extraction_label"] = object_name.strip()
                    st.session_state["selected_object"] = None
                    st.session_state["selected_objects"] = []
                    st.switch_page("pages/5_Object_Extraction.py")
import streamlit as st


PAGES = [
    ("Dashboard", "pages/1_Dashboard.py", "🏠"),
    ("Upload Image", "pages/2_Upload_Image.py", "📤"),
    ("Analysis", "pages/3_Analysis.py", "🔎"),
    ("Object Selection", "pages/4_Object_Selection.py", "🎯"),
    ("Object Extraction", "pages/5_Object_Extraction.py", "✂️"),
    ("Background Editor", "pages/6_Background_Editor.py", "🎨"),
    ("History", "pages/7_History.py", "📚"),
]


def initialize_session_state():
    defaults = {
        "uploaded_image": None,
        "image_info": None,
        "detections": [],
        "matching_detections": [],
        "detection_query": "",
        "detection_signature": None,
        "detection_error_signature": None,
        "detection_error_message": "",
        "prompt": "",
        "extraction_label": "",
        "selected_objects": [],
        "selected_object": None,
        "masks": None,
        "extracted_image": None,
        "edited_image": None,
        "processing_history": [],
        "processed_extraction_ids": [],
        "images_processed": 0,
        "objects_extracted": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: #0b1020;
            color: #eef2ff;
            --text-color: #e2e8f0;
            --secondary-text-color: #aab4ca;
            --background-color: #0b1020;
            --secondary-background-color: #11182b;
            --primary-color: #a78bfa;
        }
        [data-testid="stHeader"] {
            background: rgba(11, 16, 32, 0.9);
        }
        section[data-testid="stSidebar"] {
            background: #11182b;
            border-right: 1px solid #26314a;
        }
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        .stApp,
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stWidgetLabel"] p,
        .stApp label,
        .stApp [data-testid="stCaptionContainer"] {
            color: #e2e8f0;
        }
        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stMarkdownContainer"] li,
        .stApp [data-testid="stWidgetLabel"] p,
        .stApp label,
        .stApp [data-testid="stCaptionContainer"] {
            color: #e2e8f0 !important;
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
            color: #f8fafc !important;
        }
        .stApp p.muted-copy, .stApp .muted-copy {
            color: #aab4ca !important;
        }
        .stApp a {
            color: #c4b5fd !important;
        }
        section[data-testid="stSidebar"] [data-testid="stPageLink"] p {
            color: #e2e8f0 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #aab4ca !important;
        }
        section[data-testid="stSidebar"] h4 {
            color: #c4b5fd !important;
        }
        [data-testid="stMetricLabel"] p {
            color: #aab4ca !important;
        }
        [data-testid="stMetricValue"] {
            color: #f8fafc !important;
        }
        .stApp input, .stApp textarea {
            color: #f8fafc !important;
            background-color: #11182b !important;
        }
        .stApp [data-baseweb="select"] > div,
        .stApp [data-baseweb="input"] > div {
            color: #f8fafc;
            background-color: #11182b;
            border-color: #3b4966;
        }
        .block-container {
            max-width: 1280px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }
        .brand-mark {
            color: #a78bfa;
            font-size: 1.65rem;
            font-weight: 750;
            letter-spacing: -0.04em;
        }
        .muted-copy {
            color: #aab4ca;
        }
        div[data-testid="stMetric"] {
            background: #131c31;
            border: 1px solid #26314a;
            border-radius: 14px;
            padding: 18px 20px;
        }
        div[data-testid="stMetricLabel"] {
            color: #aab4ca;
        }
        .hero-card, .content-card {
            background: linear-gradient(135deg, #171f38 0%, #131a2d 100%);
            border: 1px solid #2a3551;
            border-radius: 18px;
            padding: 26px;
        }
        .hero-kicker {
            color: #c4b5fd;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .hero-title {
            color: #f8fafc;
            font-size: clamp(1.8rem, 3vw, 2.7rem);
            font-weight: 750;
            letter-spacing: -0.04em;
            line-height: 1.15;
            margin: 0.55rem 0;
        }
        .hero-description {
            color: #b7c1d6;
            font-size: 1.05rem;
            line-height: 1.7;
            max-width: 720px;
        }
        .feature-title {
            color: #f1f5f9;
            font-size: 1.05rem;
            font-weight: 700;
        }
        .stButton > button {
            min-height: 2.8rem;
            border-radius: 10px;
            border: 1px solid #7562e8;
            background: #6d5ce7;
            color: white;
            font-weight: 650;
        }
        .stButton > button:hover {
            background: #5e4bd4;
            border-color: #9c8cf5;
            color: white;
        }
        hr {
            border-color: #26314a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    initialize_session_state()
    render_styles()
    with st.sidebar:
        st.markdown('<div class="brand-mark">◈ Smart Vision AI</div>', unsafe_allow_html=True)
        st.caption("Object extraction workspace")
        st.divider()
        st.markdown("#### WORKSPACE")
        for label, page_path, icon in PAGES:
            st.page_link(page_path, label=label, icon=icon)
        st.divider()
        st.caption("Local object detection and segmentation")


def render_page_header(title, description, eyebrow="WORKSPACE"):
    st.caption(eyebrow)
    st.title(title)
    st.markdown(f'<p class="muted-copy">{description}</p>', unsafe_allow_html=True)


def render_dashboard():
    st.markdown(
        """
        <div class="hero-card">
          <div class="hero-kicker">Smart image segmentation</div>
          <div class="hero-title">Extract the objects<br>that matter to you.</div>
          <div class="hero-description">
            Find a common object by name, refine its cutout, then choose a clean
            background and export the result.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Images processed", st.session_state["images_processed"])
    metric_columns[1].metric("Objects extracted", st.session_state["objects_extracted"])
    metric_columns[2].metric("Workflow", "Ready to start")

    st.write("")
    st.subheader("Your workspace")
    intro, features = st.columns([1.3, 1], gap="large")
    with intro:
        st.markdown(
            """
            <div class="content-card">
              <div class="feature-title">A simple path from image to cutout</div>
              <p class="muted-copy">
                Upload a photo, describe the object you want, review the
                detected instances, then export a transparent image or try a
                new background.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Start a new project", type="primary", use_container_width=True):
            st.switch_page("pages/2_Upload_Image.py")
    with features:
        st.markdown("#### Built for precise object extraction")
        st.markdown(
            """
            - **Object search** — find common objects by name, locally.
            - **Instance selection** — choose which detected object to extract.
            - **Transparent cutouts** — keep the pixels you need.
            - **Background editing** — preview a clean final composition.
            """
        )

    st.write("")
    st.subheader("Recent activity")
    if st.session_state["processing_history"]:
        for entry in st.session_state["processing_history"][:3]:
            result_status = (
                f' · {entry["background_mode"]} background'
                if entry["background_mode"]
                else " · transparent cutout"
            )
            st.write(
                f'**{entry["object_name"]}** — {entry["image_filename"]}'
                f' · {entry["created_at"]}{result_status}'
            )
        st.page_link("pages/7_History.py", label="View processing history")
    else:
        st.info("No processing activity yet. Start a project to see it here.")

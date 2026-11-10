import tempfile
from pathlib import Path
import streamlit as st
from src.detector import ObjectDetector
from src.counter import ObjectCounter
from src.exporter import ResultExporter
st.set_page_config(
    page_title="AI Object Counting",
    page_icon="🔍",
    layout="wide"
)

st.title("AI Object Counting System")
st.write(
    "Upload an image, select object classes, "
    "and detect and count objects using YOLO."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

@st.cache_resource
def load_detector():
    return ObjectDetector()

detector = load_detector()

class_names = list(detector.model.names.values())

selected_classes = st.multiselect(
    "Select objects to count",
    options=class_names,
    default=["person", "bus"]
)
confidence = st.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Only detections with confidence at or above this value will be considered."
)
run_detection = st.button(
    "Run Detection",
    type="primary",
    use_container_width=True
)
if run_detection:

    if uploaded_file is None:
        st.warning("Please upload an image first.")

    elif not selected_classes:
        st.warning("Please select at least one object class.")

    else:
        with st.spinner("Detecting objects..."):

            with tempfile.TemporaryDirectory() as temp_dir:

                image_path = Path(temp_dir) / uploaded_file.name
                image_path.write_bytes(uploaded_file.getvalue())

                # Object detection
                result = detector.detect(
                    image_path,
                    confidence=confidence
                )

                # Object counting
                counter = ObjectCounter()

                count_result = counter.count(
                    result,
                    selected_classes
                )

                # Save annotated image
                exporter = ResultExporter()

                saved_path = exporter.save_annotated_image(
                    result,
                    image_path,
                    selected_classes
                )

        st.success("Detection completed successfully!")

        st.subheader("Detection Results")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                saved_path,
                caption="Annotated Image",
                use_container_width=True
            )

        with col2:
            st.metric("Total Objects", count_result["total"])

            for class_name, count in count_result["count"].items():
                st.metric(class_name.title(), count)
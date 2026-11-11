import pandas as pd
import tempfile
from pathlib import Path
import streamlit as st
from src.sahi_detector import SahiObjectDetector
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
        width=400
    )

@st.cache_resource
def load_detector():
    return SahiObjectDetector(
        confidence=0.25,
        slice_size=640,
        overlap=0.20,
        match_threshold=0.1
    )

detector = load_detector()

class_names = list(detector.names.values())

selected_classes = st.multiselect(
    "Select objects to count",
    options=class_names,
    default=["person", "bus"]
)
confidence = st.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.35,
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

        st.success("Image analysis completed successfully!")

        if count_result["total"] == 0:

            st.warning(
                "No selected objects were detected in this image. "
                "Try selecting different classes or adjusting "
                "the confidence threshold."
            )

        else:

            zero_classes = [
                class_name
                for class_name, count
                in count_result["count"].items()
                if count == 0
            ]

            if zero_classes:
                st.info(
                    "No detections found for: "
                    + ", ".join(zero_classes)
                )

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

        # Prepare CSV data
        csv_data = []

        for class_name, count in count_result["count"].items():
            csv_data.append({
                "filename": uploaded_file.name,
                "class_name": class_name,
                "count": count,
                "total_count": count_result["total"]
            })

        df = pd.DataFrame(csv_data)

        csv = df.to_csv(index=False)

        st.download_button(
            label="Download Counting Results (CSV)",
            data=csv,
            file_name="counting_results.csv",
            mime="text/csv",
            use_container_width=True
        )
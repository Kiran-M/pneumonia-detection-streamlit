"""Streamlit interface for the pneumonia-classification prototype."""

from pathlib import Path

import pandas as pd
import streamlit as st

from model_runtime import PneumoniaModelRuntime


APP_DIRECTORY = Path(__file__).resolve().parent
MODEL_DIRECTORY = APP_DIRECTORY / "model"


@st.cache_resource
def load_runtime():
    """Load the model once for the Streamlit process."""

    return PneumoniaModelRuntime(
        MODEL_DIRECTORY
    )


st.set_page_config(
    page_title="Chest X-ray Classification",
    page_icon="🩻",
    layout="wide",
)

st.title("Chest X-ray Classification Prototype")

st.write(
    "Upload a DICOM, PNG, JPG, or JPEG chest X-ray to "
    "view the model prediction and class probabilities."
)

st.warning(
    "Academic prototype only. This application is not a "
    "medical device and must not be used for diagnosis, "
    "treatment, or patient-management decisions."
)

st.caption(
    "Uploaded images are processed in memory and are not "
    "intentionally stored by this application."
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=[
        "dcm",
        "dicom",
        "png",
        "jpg",
        "jpeg",
    ],
    accept_multiple_files=False,
)

if uploaded_file is not None:
    try:
        runtime = load_runtime()

        prediction_result = runtime.predict(
            uploaded_file.getvalue(),
            uploaded_file.name,
        )

        image_column, result_column = st.columns(
            [1, 1]
        )

        with image_column:
            st.subheader("Model-ready image")

            st.image(
                prediction_result.display_image,
                caption=uploaded_file.name,
                clamp=True,
                use_container_width=True,
            )

        with result_column:
            st.subheader("Prediction")

            st.metric(
                label="Predicted class",
                value=prediction_result.predicted_class,
            )

            st.metric(
                label="Prediction confidence",
                value=(
                    f"{prediction_result.confidence:.2%}"
                ),
            )

            probability_dataframe = pd.DataFrame(
                [
                    {
                        "Class": class_name,
                        "Probability": probability,
                    }
                    for class_name, probability
                    in prediction_result.probabilities.items()
                ]
            )

            display_dataframe = (
                probability_dataframe.copy()
            )

            display_dataframe[
                "Probability"
            ] = display_dataframe[
                "Probability"
            ].map(
                lambda value: f"{value:.2%}"
            )

            st.subheader("Class probabilities")

            st.dataframe(
                display_dataframe,
                hide_index=True,
                use_container_width=True,
            )

            st.bar_chart(
                probability_dataframe.set_index(
                    "Class"
                ),
                y="Probability",
            )

        st.info(
            "The prediction must be interpreted together with "
            "the model limitations documented in the project report."
        )

    except Exception as application_error:
        st.error(
            "The uploaded image could not be processed. "
            f"Details: {application_error}"
        )

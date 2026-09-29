# Chest X-ray Classification Prototype

This Streamlit application loads the Validation-selected
EfficientNetB0 classifier and returns a predicted class and
probability distribution for an uploaded chest X-ray.

## Supported files

- DICOM: `.dcm` and `.dicom`
- PNG: `.png`
- JPEG: `.jpg` and `.jpeg`

The model expects a single-channel chest X-ray compatible with the
project preprocessing profile.

## Application output

- Predicted class
- Prediction confidence
- Probability for every class
- Preprocessed image used for inference

## Run without Docker

    python -m pip install -r requirements.txt
    streamlit run app.py

Open `http://localhost:8501`.

## Build and run with Docker

    docker build -t pneumonia-classifier .
    docker run --rm -p 8501:8501 pneumonia-classifier

Open `http://localhost:8501`.

## Run in GitHub Codespaces

1. Open the repository in GitHub Codespaces.
2. Build the Docker image:

       docker build -t pneumonia-classifier .

3. Run the container:

       docker run --rm -p 8501:8501 pneumonia-classifier

4. Open the **Ports** panel.
5. Set port `8501` visibility as required for the demonstration.
6. Open the forwarded URL and upload a supported image.
7. Capture the forwarded URL and successful prediction as submission evidence.

## Appropriate use

This is an academic prototype. It is not a medical device and must not be
used for diagnosis, treatment, triage, or patient-management decisions.

Uploaded images are processed in memory and are not intentionally retained
by the application.

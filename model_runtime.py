"""Model loading, image preprocessing, and inference services."""

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
import tensorflow as tf
from PIL import Image, ImageOps


@tf.keras.utils.register_keras_serializable(
    package="PneumoniaDetection"
)
class ControlledRandomAugmentation(tf.keras.layers.Layer):
    """Recreate the serialized Training-only augmentation layer."""

    def __init__(self, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed

        self.random_rotation = tf.keras.layers.RandomRotation(
            factor=0.04,
            fill_mode="reflect",
            seed=seed + 1,
        )

        self.random_translation = tf.keras.layers.RandomTranslation(
            height_factor=0.025,
            width_factor=0.025,
            fill_mode="reflect",
            seed=seed + 2,
        )

        self.random_zoom = tf.keras.layers.RandomZoom(
            height_factor=(-0.04, 0.04),
            width_factor=(-0.04, 0.04),
            fill_mode="reflect",
            seed=seed + 3,
        )

        self.random_contrast = tf.keras.layers.RandomContrast(
            factor=0.08,
            seed=seed + 4,
        )

    def build(self, input_shape):
        """Build the internal layers required by the saved model."""

        self.random_rotation.build(input_shape)
        self.random_translation.build(input_shape)
        self.random_zoom.build(input_shape)
        self.random_contrast.build(input_shape)

        super().build(input_shape)

    def apply_random_policy(self, images):
        """Apply a Training-only random augmentation policy."""

        policy_index = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=11,
            dtype=tf.int32,
            seed=self.seed,
        )

        augmented_images = tf.switch_case(
            policy_index,
            branch_fns=[
                lambda: images,
                lambda: self.random_rotation(
                    images,
                    training=True,
                ),
                lambda: self.random_translation(
                    images,
                    training=True,
                ),
                lambda: self.random_zoom(
                    images,
                    training=True,
                ),
                lambda: self.random_contrast(
                    images,
                    training=True,
                ),
                lambda: self.random_translation(
                    self.random_rotation(
                        images,
                        training=True,
                    ),
                    training=True,
                ),
                lambda: self.random_contrast(
                    self.random_rotation(
                        images,
                        training=True,
                    ),
                    training=True,
                ),
                lambda: self.random_contrast(
                    self.random_zoom(
                        images,
                        training=True,
                    ),
                    training=True,
                ),
                lambda: self.random_zoom(
                    self.random_translation(
                        images,
                        training=True,
                    ),
                    training=True,
                ),
                lambda: self.random_zoom(
                    self.random_rotation(
                        images,
                        training=True,
                    ),
                    training=True,
                ),
                lambda: self.random_contrast(
                    self.random_translation(
                        images,
                        training=True,
                    ),
                    training=True,
                ),
            ],
        )

        return tf.clip_by_value(
            augmented_images,
            0.0,
            1.0,
        )

    def call(self, images, training=None):
        """Bypass augmentation whenever the model performs inference."""

        if training is None:
            training = False

        if isinstance(training, bool):
            if training:
                return self.apply_random_policy(images)

            return images

        return tf.cond(
            tf.cast(training, tf.bool),
            lambda: self.apply_random_policy(images),
            lambda: tf.identity(images),
        )

    def get_config(self):
        """Retain the layer seed during serialization."""

        configuration = super().get_config()
        configuration.update(
            {
                "seed": self.seed,
            }
        )

        return configuration


@dataclass(frozen=True)
class PredictionResult:
    """Contain one uploaded-image inference result."""

    display_image: np.ndarray
    predicted_class: str
    confidence: float
    probabilities: dict


class PneumoniaModelRuntime:
    """Load the final model and perform dataset-aligned inference."""

    supported_extensions = {
        ".dcm",
        ".dicom",
        ".png",
        ".jpg",
        ".jpeg",
    }

    def __init__(self, model_directory):
        self.model_directory = Path(
            model_directory
        )

        self.model_path = (
            self.model_directory
            / "pneumonia_classifier.keras"
        )

        self.metadata_path = (
            self.model_directory
            / "model_metadata.json"
        )

        self.metadata = self._load_metadata()
        self.class_order = self.metadata[
            "class_order"
        ]

        self.input_shape = tuple(
            self.metadata[
                "input_shape"
            ]
        )

        self._validate_artifacts()
        self.model = self._load_model()
        self._validate_model_contract()

    def _load_metadata(self):
        """Read the deployment metadata."""

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                "Model metadata is unavailable."
            )

        with self.metadata_path.open(
            "r",
            encoding="utf-8",
        ) as metadata_file:
            return json.load(metadata_file)

    def _validate_artifacts(self):
        """Confirm that both deployment artifacts are available."""

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Serialized model is unavailable."
            )

        if self.input_shape != (320, 320, 1):
            raise ValueError(
                "Unexpected model input shape: "
                f"{self.input_shape}"
            )

        if len(self.class_order) != 3:
            raise ValueError(
                "Expected three model classes."
            )

    def _load_model(self):
        """Load the model without restoring Training configuration."""

        return tf.keras.models.load_model(
            str(self.model_path),
            compile=False,
            custom_objects={
                "ControlledRandomAugmentation": (
                    ControlledRandomAugmentation
                ),
                (
                    "PneumoniaDetection>"
                    "ControlledRandomAugmentation"
                ): ControlledRandomAugmentation,
            },
        )

    def _validate_model_contract(self):
        """Confirm that model and metadata describe the same interface."""

        if tuple(
            self.model.input_shape[1:]
        ) != self.input_shape:
            raise ValueError(
                "Model and metadata input shapes do not match."
            )

        if (
            int(self.model.output_shape[-1])
            != len(self.class_order)
        ):
            raise ValueError(
                "Model output does not match the class metadata."
            )

    @staticmethod
    def _load_dicom_image(file_bytes):
        """Read a dataset-compatible grayscale DICOM image."""

        dicom_data = pydicom.dcmread(
            io.BytesIO(file_bytes)
        )

        image = dicom_data.pixel_array.astype(
            np.float32
        )

        if image.ndim != 2:
            raise ValueError(
                "The DICOM image must be two-dimensional."
            )

        if getattr(
            dicom_data,
            "SamplesPerPixel",
            None,
        ) != 1:
            raise ValueError(
                "Only single-channel DICOM images are supported."
            )

        if getattr(
            dicom_data,
            "PhotometricInterpretation",
            None,
        ) != "MONOCHROME2":
            raise ValueError(
                "Only MONOCHROME2 DICOM images are supported."
            )

        if (
            image.min() < 0.0
            or image.max() > 255.0
        ):
            raise ValueError(
                "The DICOM pixel range is outside the "
                "0–255 range used to train the model."
            )

        return image

    @staticmethod
    def _load_standard_image(file_bytes):
        """Read a PNG or JPEG image as grayscale."""

        with Image.open(
            io.BytesIO(file_bytes)
        ) as uploaded_image:
            corrected_image = ImageOps.exif_transpose(
                uploaded_image
            )

            return np.asarray(
                corrected_image.convert("L"),
                dtype=np.float32,
            )

    def _preprocess_image(
        self,
        file_bytes,
        filename,
    ):
        """Create one model-ready 320 × 320 grayscale tensor."""

        file_extension = Path(
            filename
        ).suffix.lower()

        if (
            file_extension
            not in self.supported_extensions
        ):
            raise ValueError(
                "Supported formats are DICOM, PNG, JPG, and JPEG."
            )

        if file_extension in {
            ".dcm",
            ".dicom",
        }:
            native_image = self._load_dicom_image(
                file_bytes
            )
        else:
            native_image = self._load_standard_image(
                file_bytes
            )

        if not np.isfinite(
            native_image
        ).all():
            raise ValueError(
                "The uploaded image contains invalid pixel values."
            )

        # Match the notebook's bilinear resize and antialiasing.
        image_tensor = tf.convert_to_tensor(
            native_image,
            dtype=tf.float32,
        )

        image_tensor = tf.expand_dims(
            image_tensor,
            axis=-1,
        )

        resized_image = tf.image.resize(
            image_tensor,
            size=self.input_shape[:2],
            method="bilinear",
            antialias=True,
        )

        # Match the notebook's 0–255 clipping and 0–1 normalization.
        normalized_image = tf.clip_by_value(
            resized_image,
            0.0,
            255.0,
        ) / 255.0

        model_input = tf.expand_dims(
            normalized_image,
            axis=0,
        )

        return (
            model_input,
            normalized_image.numpy().squeeze(),
        )

    def predict(
        self,
        file_bytes,
        filename,
    ):
        """Predict one uploaded chest X-ray."""

        model_input, display_image = (
            self._preprocess_image(
                file_bytes,
                filename,
            )
        )

        predicted_probabilities = self.model(
            model_input,
            training=False,
        ).numpy()[0]

        if not np.isfinite(
            predicted_probabilities
        ).all():
            raise ValueError(
                "The model returned invalid probabilities."
            )

        if not np.isclose(
            predicted_probabilities.sum(),
            1.0,
            atol=1e-5,
        ):
            raise ValueError(
                "The model probabilities do not sum to one."
            )

        predicted_index = int(
            np.argmax(
                predicted_probabilities
            )
        )

        probability_mapping = {
            class_name: float(
                predicted_probabilities[
                    class_index
                ]
            )
            for class_index, class_name in enumerate(
                self.class_order
            )
        }

        return PredictionResult(
            display_image=display_image,
            predicted_class=self.class_order[
                predicted_index
            ],
            confidence=float(
                predicted_probabilities[
                    predicted_index
                ]
            ),
            probabilities=probability_mapping,
        )

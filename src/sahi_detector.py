
from pathlib import Path

import cv2
import torch

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics.engine.results import Results


class SahiObjectDetector:

    def __init__(
        self,
        model_path="yolo11n.pt",
        names=None,
        confidence=0.25,
        slice_size=640,
        overlap=0.20,
        match_threshold=0.1
    ):

        self.names = names

        self.confidence = confidence
        self.slice_size = slice_size
        self.overlap = overlap
        self.match_threshold = match_threshold

        self.device = (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=confidence,
            device=self.device,
            image_size=1536
        )
        if self.names is None:
            self.names = {
                int(class_id): class_name
                for class_id, class_name
                in self.detection_model.category_mapping.items()
            }

    def detect(self, image_path, confidence=None):

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found at: {image_path}"
            )

        original_image = cv2.imread(str(image_path))

        if original_image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        # Run sliced inference
        if confidence is not None:
            self.detection_model.confidence_threshold = confidence
        sahi_result = get_sliced_prediction(
            image=str(image_path),
            detection_model=self.detection_model,
            slice_height=self.slice_size,
            slice_width=self.slice_size,
            overlap_height_ratio=self.overlap,
            overlap_width_ratio=self.overlap,
            postprocess_type="NMS",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=self.match_threshold,
            verbose=0
        )

        # Convert SAHI detections into Ultralytics format
        boxes = []

        for prediction in sahi_result.object_prediction_list:

            x1, y1, x2, y2 = prediction.bbox.to_xyxy()

            confidence = prediction.score.value
            class_id = int(prediction.category.id)

            boxes.append([
                x1,
                y1,
                x2,
                y2,
                confidence,
                class_id
            ])

        if boxes:
            boxes_tensor = torch.tensor(
                boxes,
                dtype=torch.float32
            )
        else:
            boxes_tensor = torch.empty((0, 6), dtype=torch.float32)

        # Create a standard Ultralytics Results object
        result = Results(
            orig_img=original_image,
            path=str(image_path),
            names=self.names,
            boxes=boxes_tensor
        )

        return result

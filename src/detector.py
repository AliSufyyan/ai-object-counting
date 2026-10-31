from pathlib import Path
from ultralytics  import YOLO

class ObjectDetector:
    def __init__(self,model_path='yolo11n.pt'):
        self.model=YOLO(model_path)


    def detect(self,image_path,confidence=0.25):
        image_path=Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError("Image not found at ", image_path)

        result=self.model.predict(
            source=str(image_path),
            conf=confidence,
            verbose=False
        )

        return result[0]
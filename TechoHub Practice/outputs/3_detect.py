
import sys
print("Python executable:", sys.executable)
from ultralytics import YOLO
from pathlib import Path

image_path=Path('images/test.jpg')


if not image_path.exists():
    raise FileNotFoundError (
        f"Image not found at {image_path}"
    )


model=YOLO('yolo11n.pt')

results=model.predict(
    source=str(image_path),
    conf=0.25,
    verbose=False
)

result=results[0]

for box in result.boxes:
    class_id=int(box.cls.item())
    class_name=result.names[class_id]
    confidence=float(box.conf.item())
    coordinates=box.xyxy[0].tolist()

    print("Class: ",class_name)
    print('Confidence: ',round(confidence,3))
    print('Box', coordinates)
    print('-------------------')

output_path=Path('outputs')
output_path.mkdir(exist_ok=True)

annotated_img=result.plot()

import cv2

cv2.imwrite(
    str(output_path/"annotated.jpg"),
    annotated_img
)
print('Detection completed')




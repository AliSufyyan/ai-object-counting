import cv2
from ultralytics import YOLO 
from pathlib import Path

image_path=Path('images/test.jpg')

if not image_path.exists():
    raise FileNotFoundError(f"File not Fount {image_path}")


model=YOLO('yolo11n.pt')

threshold=0.75

results=model.predict(
    source=str(image_path),
    conf=threshold,
    verbose=False
)
counts={}
result=results[0]

for box in result.boxes:
    class_id=int(box.cls.item())
    class_name=result.names[class_id]
    confidence=float(box.conf.item())
    counts[class_name]=counts.get(class_name,0)+1

    print(f"{class_name} Confidence: {confidence:.3f}")


print('\n Claswise Count')

for class_name,count in sorted(counts.items()):
    print(f"{class_name} : {count}")


print(f"Total Count : {sum(counts.values())}")
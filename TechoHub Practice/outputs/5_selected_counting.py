import cv2
from ultralytics import YOLO
from pathlib import Path

image_path=Path('images/bus.jpg')

if not image_path.exists():
    raise FileNotFoundError(f"File not found at {image_path}")

model=YOLO('yolo11n.pt')

results=model.predict(
    source=str(image_path),
    conf=0.25,
    verbose=False
)

result=results[0]
selected_class={'person','bus'}
counts = {class_name: 0 for class_name in selected_class}
for box in result.boxes:
    class_id=int(box.cls.item())
    class_name=result.names[class_id]
    confidence=float(box.conf.item())
    coordinates=box.xyxy[0].tolist()
    if class_name not in selected_class:
        continue

    counts[class_name]=counts.get(class_name,0)+1
    print(f'{class_name} : Confidence{confidence:.3f}')
    print(f'{class_name} Coordinates: {coordinates}')
    


for class_name,count in sorted(counts.items()):
    print(f"{class_name} Count : {count}")


print('\n Total count',sum(counts.values()))

annotated=result.plot()
import cv2

output=Path('outputs')
output.mkdir(exist_ok=True,parents=True)
cv2.imwrite(
    str(output/'bus_annotated.jpg'),
    annotated
)

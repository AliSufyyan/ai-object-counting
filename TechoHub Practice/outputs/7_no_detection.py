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


for class_name,count in sorted(counts.items()):
    print(f"{class_name} Count : {count}")


total=sum(counts.values())
print('\n Total count',total)
if total > 0:
    print("Selected objects detected successfully.")
else:
    print("No selected objects detected.")

from datetime import datetime
import csv 

filename=image_path.name

timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')

output_dir=Path('CSVs')
output_dir.mkdir(exist_ok=True,parents=True)

csv_path=output_dir/'counting_result.csv'

fieldnames=[
    'timestamp',
    'filename',
    'person',
    'bus',
    'total_count'
]

file_exists_and_has_data=(
    csv_path.exists() and csv_path.stat().st_size>0
)

with open(csv_path, 'a', newline='',encoding='utf-8') as file:
    writer=csv.DictWriter(file,fieldnames=fieldnames)

    if not file_exists_and_has_data:
        writer.writeheader()

    writer.writerow({
        'timestamp':timestamp,
        'filename':filename,
        'person':counts.get('person',0),
        'bus':counts.get('bus',0),
        'total_count':sum(counts.values())
    })


print(f"CSV Report saved to: {csv_path}")
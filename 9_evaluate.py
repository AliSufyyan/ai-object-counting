from pathlib import Path
test_folder=Path('images/test')
image_files=list(test_folder.glob('*.jpg'))
for image in image_files:
    print(image)

from ultralytics import YOLO
model=YOLO('yolo11n.pt')
print('Model laoded successfully')

results=model.predict(
    source=str(test_folder),
    conf=0.25,
    verbose=False
)
i=0
selected_class={'person','bus'}

fieldsname=[
    'timestamp',
    'filename',
    'person',
    'bus',
    'count'
]
import csv
from datetime import datetime
for result in  results:
    counts={class_name:0 for class_name in selected_class}
    print(f'For image {i+1}')
    for box in result.boxes:
        class_id=int(box.cls.item())
        class_name=result.names[class_id]
        confidence=float(box.conf.item())
        coordiates=box.xyxy[0].tolist()

        counts[class_name]=counts.get(class_name,0)+1


    for class_name,count in sorted(counts.items()):
        print(f'{class_name}---->{count}')
    print('\nTotal Count',sum(counts.values()))
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    import cv2
    annotatd=result.plot
    output=Path('outputs')
    output.mkdir(exist_ok=True,parent=True)
    annotated_image='Image'+str(i)
    cv2.imwrite(
        str(output/annotated_image),
        annotatd
    )
    filename=image_files[i]

    output_dir=Path('CSVs')
    output_dir.mkdir(parents=True,exist_ok=True)
    csv_path=output_dir/'Multi_image.csv'

    file_exists_has_no_data=(
        csv_path.exists() and csv_path.stat().st_size>0
    )

    with open(csv_path,'a',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=fieldsname)

        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if not file_exists_has_no_data:
            writer.writeheader()

        writer.writerow({
            'timestamp':timestamp,
            'filename':filename,
            'person':counts.get('person',0),
            'bus':counts.get('bus',0),
            'count':sum(counts.values())
        }
        )
        print(f'CSV file appended at {csv_path}')

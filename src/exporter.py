from pathlib import Path 
from datetime import datetime
import cv2
import csv
class ResultExporter:
    def __init__(self,object_dir='outputs/annotated'):
        self.output_dir=Path(object_dir)
        self.output_dir.mkdir(parents=True,exist_ok=True)

    def save_annotated_image(self,result,image_path):
        image_path=Path(image_path)

        annotated_image=result.plot()

        timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')

        output_filename=(
            f"{image_path.stem}_{timestamp}_annotated.jpg"
        )

        output_path=self.output_dir / output_filename

        saved=cv2.imwrite(str(output_path),annotated_image)

        if not saved:
            raise OSError(f"Could not saved {output_path}")

        return output_path
    def export_count(self,image_path,count_result):
        csv_path=Path('outputs/counting_result.csv')
        csv_path.parent.mkdir(parents=True,exist_ok=True)

        file_exists=csv_path.exists() and csv_path.stat().st_size>0

        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        fieldnames=[
            'filename',
            'timestamp',
            'class_name',
            'count',
            'total_count'
        ]
        with open(csv_path,mode='a',newline='',encoding='utf-8') as file:
            writer=csv.DictWriter(file,fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()


            for class_name,count in count_result['count'].items():
                writer.writerow({
                    'filename':Path(image_path).name,
                    'timestamp':timestamp,
                    'class_name':class_name,
                    'count':count,
                    'total_count':count_result['total']
                })
        return csv_path
            

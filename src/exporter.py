from pathlib import Path 
from datetime import datetime
import cv2

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
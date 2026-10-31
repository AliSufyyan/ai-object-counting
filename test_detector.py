from src.detector import ObjectDetector
from src.counter import ObjectCounter
from src.exporter import ResultExporter
detector=ObjectDetector()
counter=ObjectCounter()
exporter=ResultExporter()
result=detector.detect("images/test/bus.jpg")

selected_class=['person','bus']

count_results=counter.count(result,selected_class)

saved_path=exporter.save_annotated_image(result,"images/test/bus.jpg")

print("Image Detected")
print("Selected Class count", count_results['count'])
print('Total Count',count_results['total'])
print("Annotated Image Saved:", saved_path)
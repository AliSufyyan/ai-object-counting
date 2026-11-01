from src.detector import ObjectDetector
from src.counter import ObjectCounter
from src.exporter import ResultExporter
import cv2,csv
from datetime import datetime
from pathlib import Path 
from ultralytics import YOLO  

image_folder=Path('images/test')

detector=ObjectDetector()
counter=ObjectCounter()
exporter=ResultExporter()

selected_class=['person','bus']
for image_path in image_folder.glob("*.jpg"):
    result=detector.detect(image_path)
    count_results=counter.count(result,selected_class)
    saved_path=exporter.save_annotated_image(result,image_path)
    csv_path=exporter.export_count(image_path,count_results)
    
    


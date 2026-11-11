
from src.detector import ObjectDetector
from src.sahi_detector import SahiObjectDetector

image_path = "images/test/image_03.jpg"


standard_detector = ObjectDetector()


sahi_detector = SahiObjectDetector(
    names=standard_detector.model.names
)


result = sahi_detector.detect(image_path)

print("Detection type:", type(result))
print("Total detections:", len(result.boxes))


for box in result.boxes:
    class_id = int(box.cls.item())
    class_name = result.names[class_id]

    print(class_name)


result.save("outputs/sahi_test.jpg")

print("Test completed!")

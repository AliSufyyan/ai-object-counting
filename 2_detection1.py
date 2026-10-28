from ultralytics import YOLO

model=YOLO('yolo11n.pt')

print("Model names", len(model.names))
print('Person Class ID:', 0)
print("Person class name: ",model.names[0])
print('Car class name: ',model.names[2])


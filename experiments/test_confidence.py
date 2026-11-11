
from pathlib import Path
from ultralytics import YOLO
import csv

# Paths
image_folder = Path("images/test")
output_file = Path("experiments/experiment_results.csv")

# Load model once
model = YOLO("yolo11n.pt")

# Three independent configurations
experiments = [
    {"name": "baseline", "confidence": 0.25, "imgsz": 640},
    {"name": "low_confidence", "confidence": 0.10, "imgsz": 640},
    {"name": "high_resolution", "confidence": 0.25, "imgsz": 1280},
]

results_data = []

# Process every test image
for image_path in sorted(image_folder.glob("*.jpg")):

    print(f"\nProcessing: {image_path.name}")

    for experiment in experiments:

        result = model.predict(
            source=str(image_path),
            conf=experiment["confidence"],
            imgsz=experiment["imgsz"],
            verbose=False
        )[0]

        # Initialize counts for all supported classes
        class_counts = {
            class_name: 0
            for class_name in result.names.values()
        }

        # Count detections
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = result.names[class_id]

            class_counts[class_name] += 1

        total = sum(class_counts.values())

        # Store results for every class
        for class_name, count in class_counts.items():
            results_data.append({
                "filename": image_path.name,
                "experiment": experiment["name"],
                "confidence": experiment["confidence"],
                "imgsz": experiment["imgsz"],
                "class_name": class_name,
                "count": count,
                "total_count": total
            })

        print(
            f"{experiment['name']}: "
            f"Total={total}, "
            f"Persons={class_counts['person']}, "
            f"Buses={class_counts['bus']}"
        )

# Export experiment results
fieldnames = [
    "filename",
    "experiment",
    "confidence",
    "imgsz",
    "class_name",
    "count",
    "total_count"
]

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results_data)

print(f"\nExperiment completed. Results saved to: {output_file}")

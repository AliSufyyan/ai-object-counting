
from sahi.utils.cv import visualize_object_predictions
from pathlib import Path
import csv
import cv2
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

IMAGE_FOLDER = Path("images/test")
GROUND_TRUTH_FILE = Path("data/ground_truth.csv")
BASELINE_FILE = Path("experiments/experiment_results.csv")
OUTPUT_FILE = Path("experiments/tiling_results.csv")

MODEL_PATH = "yolo11n.pt"

CONFIDENCE = 0.25
SLICE_SIZE = 512
OVERLAP = 0.20

MATCH_THRESHOLDS = [0.3, 0.5, 0.7]
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


# --------------------------------------------------
# 2. Load the YOLO model through SAHI
# --------------------------------------------------

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=MODEL_PATH,
    confidence_threshold=CONFIDENCE,
    device=device
)


# --------------------------------------------------
# 3. Load ground truth
# --------------------------------------------------

ground_truth = {}

with open(GROUND_TRUTH_FILE, newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        filename = row["filename"]

        ground_truth[filename] = {
            "person": int(row["person"]),
            "bus": int(row["bus"])
        }


# --------------------------------------------------
# 4. Load previous baseline results
# --------------------------------------------------

baseline = {}

with open(BASELINE_FILE, newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if row["experiment"] == "baseline":
            key = (row["filename"], row["class_name"])
            baseline[key] = int(row["count"])

    # --------------------------------------------------
    # 5. Run threshold experiments
    # --------------------------------------------------

    image_paths = sorted(IMAGE_FOLDER.glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError("No JPG images found in images/test")

    tiling_results = []

    comparison = {
        threshold: {
            "person": 0,
            "bus": 0
        }
        for threshold in MATCH_THRESHOLDS
    }

    for threshold in MATCH_THRESHOLDS:

        print(f"\n========== THRESHOLD: {threshold} ==========")

        for image_path in image_paths:

            print(f"\nProcessing: {image_path.name}")

            result = get_sliced_prediction(
                image=str(image_path),
                detection_model=detection_model,
                slice_height=SLICE_SIZE,
                slice_width=SLICE_SIZE,
                overlap_height_ratio=OVERLAP,
                overlap_width_ratio=OVERLAP,
                postprocess_type="GREEDYNMM",
                postprocess_match_metric="IOU",
                postprocess_match_threshold=threshold,
                verbose=0
            )
            image = cv2.imread(str(image_path))

            visualize_object_predictions(
                image=image,
                object_prediction_list=result.object_prediction_list,
                output_dir="outputs/tiling_annotated",
                file_name=f"{image_path.stem}_threshold_{threshold}",
                export_format="jpg"
            )
            # Count detections by class
            class_counts = {}

            for prediction in result.object_prediction_list:
                class_name = prediction.category.name

                class_counts[class_name] = (
                    class_counts.get(class_name, 0) + 1
                )

            for class_name in ["person", "bus"]:
                class_counts.setdefault(class_name, 0)

            total = sum(class_counts.values())

            print(
                f"Total={total}, "
                f"Persons={class_counts['person']}, "
                f"Buses={class_counts['bus']}"
            )

            # Save all detected classes
            for class_name, count in class_counts.items():

                tiling_results.append({
                    "filename": image_path.name,
                    "threshold": threshold,
                    "class_name": class_name,
                    "count": count,
                    "total_count": total
                })

            # Calculate absolute errors
            for class_name in ["person", "bus"]:

                actual = ground_truth[image_path.name][class_name]

                error = abs(
                    actual - class_counts[class_name]
                )

                comparison[threshold][class_name] += error


    # --------------------------------------------------
    # 6. Export experiment results
    # --------------------------------------------------

    fieldnames = [
        "filename",
        "threshold",
        "class_name",
        "count",
        "total_count"
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tiling_results)


    # --------------------------------------------------
    # 7. Compare MAE
    # --------------------------------------------------

    print("\n========== FINAL MAE COMPARISON ==========")

    for threshold, values in comparison.items():

        image_count = len(image_paths)

        person_mae = values["person"] / image_count
        bus_mae = values["bus"] / image_count

        print(f"\nThreshold: {threshold}")
        print(f"Person MAE: {person_mae:.2f}")
        print(f"Bus MAE: {bus_mae:.2f}")

    print(f"\nResults saved to: {OUTPUT_FILE}")

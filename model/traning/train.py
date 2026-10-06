from pathlib import Path
from ultralytics import YOLO


BASE_DIR = Path(
    r"Your\Path\To\Your\Project"
)

DATASET_DIR = BASE_DIR / "MWPD"
DATA_YAML = DATASET_DIR / "data.yaml"


DATA_YAML.write_text(
    f"""path: {DATASET_DIR.as_posix()}
train: train/images
val: valid/images
test: test/images

names:
  0: pothole
""",
    encoding="utf-8"
)


MODEL_NAME = "yolov8s.pt"



def main():

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        seed=42,
        amp=True,
        patience=30,
        plots=True,
        save=True,
        project=str(BASE_DIR / "runs" / "detect"),
        name="yolov8s_mwpd_exp1",
        exist_ok=True
    )

    print("\nTraining completed.")

    print("\nBest model:")
    print(
        BASE_DIR
        / "runs"
        / "detect"
        / "yolov8s_mwpd_exp1"
        / "weights"
        / "best.pt"
    )


if __name__ == "__main__":
    main()
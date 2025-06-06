import os
from ultralytics import YOLO
import cv2
import numpy as np
import json
import pandas as pd
from pathlib import Path


def extract_bboxes(mask):
    """
    extract unique segmentation mask from label and use (x_min, y_min, x_max, y_max) as bbox coordinate
    :param mask: label image from dataset
    :return: bboxes coordinate (x_min, y_min, x_max, y_max)
    """
    # extract every unique instance in the mask
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids != 0]  # Skip background
    bboxes = []

    # create binary mask for each instance
    for instance_id in instance_ids:
        binary_mask = (mask == instance_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # find bounding box of contour
        for contour in contours:
            if len(contour) >= 3:
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append((x, y, x + w, y + h))  # (x_min, y_min, x_max, y_max)

    return bboxes


def save_yolo_bboxes(bboxes, img_shape, output_txt_path, class_id=0):
    """
    Save txt file in YOLO format <class_id> <x_center> <y_center> <width> <height>
    :param bboxes: bboxes coordinate
    :param img_shape:
    :param output_txt_path:
    :param class_id: default 0 for strawberry
    """
    h, w = img_shape
    lines = []

    for x_min, y_min, x_max, y_max in bboxes:
        cx = (x_min + x_max) / 2 / w
        cy = (y_min + y_max) / 2 / h
        bw = (x_max - x_min) / w
        bh = (y_max - y_min) / h
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    with open(output_txt_path, 'w') as f:
        f.write("\n".join(lines))


def evaluate_on_test_set(weight_path, eval_output_path=None):
    """
    Evaluate model on test set and save metrics including precision, recall,
    mAP50, mAP50-95, and fitness to output in json
    :param weight_path: path to model
    :param eval_output_path: path to output
    :return:
    """
    if eval_output_path is None:
        os.makedirs("eval_json", exist_ok=True)
        eval_output_path = "eval_json"

    model = YOLO(weight_path)

    metrics = model.val(data="../config/test_dataset.yaml", split="test", plots=True, project=eval_output_path)
    if hasattr(metrics, 'results_dict'):
        serializable_metrics = metrics.results_dict
    else:
        serializable_metrics = {k: float(v) for k, v in metrics.items()}

    with open(f"{os.path.join(eval_output_path, 'eval')}.json", "w") as f:
        json.dump(serializable_metrics, f, indent=4)


def summary_grid_to_csv(base_dir, output_path=None):
    """
    Iterate through grid search project and save losses and evaluation metrics into
    a csv file
    :param base_dir: base grid search project directory
    :param output_path: path to output csv file
    :return:
    """
    base_dir = Path(base_dir)
    if output_path is None:
        output_path = os.path.join(base_dir, "grid_summary.csv")
    summary = []
    for folder in base_dir.iterdir():
        result_file = folder / "results.csv"
        if result_file.exists():
            df = pd.read_csv(result_file)
            # print(df.columns)

            row = {
                "run": folder.name,
                "mAP50": df.iloc[-1]["metrics/mAP50(B)"],
                "mAP50-95": df.iloc[-1]["metrics/mAP50-95(B)"],
                "precision": df.iloc[-1]["metrics/precision(B)"],
                "recall": df.iloc[-1]["metrics/recall(B)"],
                "train/box_loss": df.iloc[-1]["train/box_loss"],
                "val/box_loss": df.iloc[-1]["val/box_loss"],
                "box_loss_diff": abs(df.iloc[-1]["train/box_loss"] - df.iloc[-1]["val/box_loss"]),
                "train/dfl_loss": df.iloc[-1]["train/dfl_loss"],
                "val/dfl_loss": df.iloc[-1]["val/dfl_loss"],
                "dfl_loss_diff": abs(df.iloc[-1]["train/dfl_loss"] - df.iloc[-1]["val/dfl_loss"])
            }
            summary.append(row)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_path, index=False)
    print(f"Saved summary to {base_dir / 'grid_summary.csv'}")

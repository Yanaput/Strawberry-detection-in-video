from ultralytics import YOLO
import itertools
import os
from util import summary_grid_to_csv


def grid_search():
    lr0_list = [0.002, 0.001]
    weight_decay_list = [0.001, 0.0005]
    optimizer_list = ["SGD", "AdamW"]
    imgsz_list = [640]

    for lr0, wd, opt, imgs in itertools.product(lr0_list, weight_decay_list, optimizer_list, imgsz_list):
        run_name = f"lr{lr0}_wd{wd}_{opt}"
        if os.path.exists(f"gridsearch/{run_name}"):
            print(f"{run_name} existed")
            continue

        print(f"Training with {run_name}")
        model = YOLO("pre_trained/yolo11s.pt")
        model.train(
            data="../config/dataset.yaml",
            epochs=30,
            imgsz=imgs,
            lr0=lr0,
            weight_decay=wd,
            optimizer=opt,
            project="gridsearch",
            name=run_name,
            device=0,
            val=True,
            plots=True,
        )


if __name__ == '__main__':
    grid_search()
    summary_grid_to_csv("./gridsearch")

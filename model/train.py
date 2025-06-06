from ultralytics import YOLO


def main():
    model = YOLO("pre_trained/yolo11s.pt")
    model.train(
        patience=33,
        project=f"SGD_lr0{0.002}_wd{0.0005}_augment",
        pretrained=True,
        data="../config/dataset.yaml",
        optimizer="SGD",
        epochs=100,
        imgsz=768,
        lr0=0.002,
        weight_decay=0.0005,
        degrees=10.0,
        translate=0.1,
        scale=0.2,
        shear=2.0,
        perspective=0.0,
        fliplr=0.5,
        mosaic=1.0,
        device=0,
        plots=True,
        val=True
    )


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main()

from util import evaluate_on_test_set


def main():
    evaluate_on_test_set("./SGD_lr00.002_wd0.0005_augment/train/weights/best.pt", "./SGD_lr00.002_wd0.0005_augment")


if __name__ == '__main__':
    main()

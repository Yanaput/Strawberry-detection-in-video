import os
import cv2
from util import extract_bboxes, save_yolo_bboxes


def main():
    base_dataset_dir = "../data/StrawDI_Db1/"
    dataset_splits = ["test", "train", "val"]

    '''
        extract polygons from label, which is grey-scaled mask. 
        Since YOLO expecting txt files containing bounding box coordinate in format 
        <class_id> <x_center> <y_center> <width> <height>, I extract polygons in image 
        and use x min/max and y min/max as bounding box coordinate
    '''
    for split_set in dataset_splits:
        label_dir = os.path.join(base_dataset_dir, split_set, "masks")
        img_dir = os.path.join(base_dataset_dir, split_set, "images")

        output_label_dir = os.path.join(base_dataset_dir, split_set, "labels")
        os.makedirs(output_label_dir, exist_ok=True)

        for filename in os.listdir(label_dir):
            output_filename = os.path.join(output_label_dir, filename.replace('.png', '.txt'))
            if not filename.endswith('.png'):
                continue
            mask = cv2.imread(os.path.join(label_dir, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(os.path.join(img_dir, filename))
            bboxes = extract_bboxes(mask)
            save_yolo_bboxes(bboxes, img.shape[:2], output_filename)


if __name__ == '__main__':
    main()


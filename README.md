### Requirement
- Detect and locate strawberries in a static
image.
- Apply video processing techniques ti count number of strawberries in video

### Dataset
**_[StrawDI](https://strawdi.github.io/)_**, a dataset contains 3,100 images of strawberries and corresponding ground 
truth. Ground truth images contain segmentation of all the strawberries in the image. Ground truth images are grayscale 
images, where 0 represents background and others represent index of strawberry in the image.

### Approach 
Since the ground truth images are segmentation of strawberries, but the final goal is to detect and count the number of 
strawberry in the video, I was a bit hesitate whether I should approach with segmentation task or detection task.
However, I choose to focus on the final goal and considered that since the model only need to **_detect and count_**, 
detection with bounding box would be sufficient.

### YOLO, pretrained model
To reduce training and configuration time I chose [Ultralytics YOLO](https://docs.ultralytics.com/) (You Only Look Once)
as my pretrained model. 

YOLO is a fast, accurate, easy to use, and widely adopted in many computer vision tasks. YOLO offers various vary 
on versions and task. I chose YOLO11s due to its lightweight architecture, which suited with my hardware limitation 
and allows for faster training.

### Data preprocessing
First of all YOLO prefers data directory structured as
```commandline
dataset
├──test/
│   └──images
│   └──labels   // contains txt file
├──train/
├──val/

```
Therefore, I rename the dataset as it preferred. Where img -> images, but label -> masks to prevent my own confusion.

YOLO also require labels to be in .txt extension. And for detection task each line inside each label should be in format 
```commandline
<class_id> <x_center> <y_center> <width> <height>
```
Which tells class of the object, center of the object, width and height of the objects' bounding box.

I applied binary mask in each segment instances in labels the use [findContours()](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0) 
from OpenCV to extract outlines that follow the edges of objects and use [boundingRect()](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga103fcbda2f540f3ef1c042d6a9b35ac7) 
to get bounding box each strawberry segment. Then save the corner for bounding boxes in txt file.

### Grid Search
In order to find parameters suited for training strawberry class, I perform grid search train on parameters including initial 
learning rate, weight decay, and optimizer
```commandline
| lr0          | 0.002 | 0.001  |
| weight_decay | 0.001 | 0.0005 |
| optimizer    | SGD   | AdamW  |
```
with epoch=30 and imgsize=640.

I then chose the parameters set based on the accuracy metrics and losses. 

The set parameters I chose is
```commandline
| lr0   | weight_decay | optimizer |
|-------|--------------|-----------|
| 0.002 | 0.0005       | SGD       |
```
since it has the highest mAP50-95 and F1 score. Along with difference between train and validation losses are relatively 
low, meaning there's no sign of overfitting.

### Training
Beside the parameters set from gird search, on the training I increase images resolution from 640 to 728, increase 
to 100 epochs. Also, to increase then generalization of the model I augmented data to increase the diversity in the data.


### Video inference
YOLO provided object tracking feature, which allow user to choose tracker type either botsort or bytetrack and configure other
parameters.

To count the number of unique strawberries, I use the track_ids and confidences returned by YOLO's tracking results in each frame.

I use set to store of unique track_id and use confidences to filter the object 
if
- the track_id is valid (i.e., not None or -1),
- And the confidence > 0.6

Then increment the count only once per unique track_id.

Download demo video [here](https://github.com/Yanaput/Strawberry-detection-in-video/blob/master/inference/output_video.mp4)

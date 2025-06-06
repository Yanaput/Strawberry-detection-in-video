import cv2
from ultralytics import YOLO

model = YOLO("../model/SGD_lr00.002_wd0.0005_augment/train/weights/best.pt")

cap = cv2.VideoCapture("test.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


# Save output with detection overlays
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

counted_ids = set()
total_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(source=frame, verbose=False, persist=True, tracker='../config/bytetrack.yaml')[0]
    boxes = results.boxes.xyxy.cpu()
    track_ids = results.boxes.id
    confidences = results.boxes.conf.cpu()

    if track_ids is None:
        track_ids = [-1] * len(boxes)  # -1 if invalid
    else:
        track_ids = track_ids.int().cpu().tolist()

    for box, track_id, conf in zip(boxes, track_ids, confidences):
        if track_id == -1 or conf < 0.6:
            continue  # skip if invalid or low confidence

        if track_id not in counted_ids:
            counted_ids.add(track_id)
            total_count += 1
            print(f"Counted {track_id}")

        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2

        cy = (y1 + y2) // 2
        cv2.putText(frame, f"ID# {track_id} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255))
        cv2.circle(frame, (cx, cy), 4, (255, 255, 0), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f"Count: {total_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    out.write(frame)
    cv2.imshow("Strawberry Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print(f"Count: {total_count}")

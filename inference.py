import cv2
from ultralytics import solutions
import argparse
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "./model/SGD_lr00.002_wd0.0005_augment/train/weights/best.pt"
DEFAULT_VIDEO = SCRIPT_DIR / "./test.mp4"
DEFAULT_TRACKER = SCRIPT_DIR / "./config/bytetrack.yaml"


def infer(
        model_path:str,
        video_path:str,
        tracker_config:str
):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    region_points = [(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)]

    counter = solutions.ObjectCounter(
        show=True,
        region=region_points,
        model=model_path,
        tracker=tracker_config,
    )

    out_path = SCRIPT_DIR / "inference_output" / "output_video.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    video_writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = counter.process(frame)
        composed = results.plot_im
        cv2.putText(composed, f"Count: {len(counter.counted_ids)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        video_writer.write(results.plot_im)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Count: {len(counter.counted_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run object counting on a video.")
    parser.add_argument("--model_path", nargs="?",
                        default=DEFAULT_MODEL,
                        help="Path to YOLO model .pt (default: %(default)s)",
                        type=str
    )
    parser.add_argument("--video_path", nargs="?",
                        default=DEFAULT_VIDEO,
                        help="Path to input video (default: %(default)s)",
                        type=str
    )
    parser.add_argument("--tracker_config", nargs="?",
                        default=DEFAULT_TRACKER,
                        help="Path to input video (default: %(default)s)",
                        type=str
    )
    args = parser.parse_args()

    infer(args.model_path, args.video_path, args.tracker_config)

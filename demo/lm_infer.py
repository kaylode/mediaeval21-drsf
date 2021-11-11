import argparse

import cv2
from estimators.landmarks import LandmarkEstimator
from helper.common import draw_points, plot_box, video_metadata
from tqdm import tqdm
from pathlib import Path

parser = argparse.ArgumentParser("Video De-identification")
parser.add_argument(
    "--video_path", "-i", type=str, required=True, help="Video input path"
)
parser.add_argument("--output_path", "-o", type=str, help="Video output path")
parser.add_argument(
    "--detector", "-d", type=str, default="retinaface", help="face detector"
)
parser.add_argument("--alignment", "-a", type=str, default="fan", help="face alignment")

args = parser.parse_args()
BATCH_SIZE = 2
VIDEO_PATH, OUTPUT_PATH = Path(args.video_path), Path(args.output_path)
assert VIDEO_PATH.exists(), f"{VIDEO_PATH} does not exist"
assert OUTPUT_PATH.exists(), f"{OUTPUT_PATH} does not exist"
VIDEO_NAME = VIDEO_PATH.stem
OUTPUT_PATH = OUTPUT_PATH / f"{VIDEO_NAME}_result.avi"
print(f"Result video is saved at {OUTPUT_PATH}")

if __name__ == "__main__":
    e = LandmarkEstimator.from_name(det_name=args.detector, align_name=args.alignment)

    CAP = cv2.VideoCapture(args.video_path)
    w, h, fps, total_frames = video_metadata(CAP)

    writter = cv2.VideoWriter(
        str(OUTPUT_PATH), cv2.VideoWriter_fourcc("M", "J", "P", "G"), fps, (w, h)
    )

    batch = []
    frame_id = 0
    with tqdm(total=total_frames) as pbar:
        while CAP.isOpened():
            ret, frame = CAP.read()
            if not ret:
                if frame_id < total_frames:
                    print("not finished")
                break

            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch.append(cv2_image)
            if (len(batch)) % BATCH_SIZE == 0 or frame_id == total_frames - 1:

                face_boxes, landmarks = e.detect_faces(batch)
                for im, face_box, landmark in zip(batch, face_boxes, landmarks):
                    image = im.copy()
                    plot_box(image, face_box)
                    draw_points(image, landmark)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    writter.write(image)

                batch = []

            frame_id += 1
            pbar.update(1)


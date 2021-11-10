import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from attack.attacker.full_attacker import FullAttacker
from attack.deid import Pixelate, Blur
from models.face_align.fan import FANAlignment
from models.face_align.models.face_alignment.utils import crop
from models.face_det.retinaface import RetinaFaceDetector
from attack.attacker import generate_tensors
import argparse

parser = argparse.ArgumentParser("Video De-identification")
parser.add_argument(
    "--video_path", "-i", type=str, required=True, help="Video input path"
)
parser.add_argument("--output_path", "-o", type=str, help="Video output path")
parser.add_argument(
    "--detector", "-d", type=str, default="retinaface", help="Victim detector"
)
parser.add_argument(
    "--alignment", "-a", type=str, default="fan", help="Victim alignment"
)
parser.add_argument("--algorithm", "-g", type=str, default="rmsprop", help="Algorithm")
parser.add_argument(
    "--deid", "-m", type=str, default="pixelate", help="De-identification method"
)
parser.add_argument(
    "--num_iters", type=int, default=30, help="Number of iterations to attack"
)


def plot_box(image, pred, figsize=(10, 10)):
    pred = np.squeeze(pred).tolist()
    box = [int(i) for i in pred]
    x, y, x2, y2 = box
    cv2.rectangle(image, (x, y), (x2, y2), color=(255, 0, 0), thickness=2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    plt.axis("off")
    plt.close()
    return fig


def _convert_pt(point: np.ndarray):
    return tuple(np.round(point).astype(np.int).tolist())


def draw_points(image, points, color=(255, 0, 0), size=3, figsize=(10, 10)) -> None:
    assert image is not None
    assert points.shape[1] == 2
    for pt in points:
        pt = _convert_pt(pt)
        cv2.circle(image, pt, size, color, cv2.FILLED)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)

    plt.axis("off")
    plt.close()
    return fig


def doit(batch, det_model, align_model, attacker, deid_fn):

    # Generate truth bboxes
    det_norm = det_model.preprocess(batch)
    det_norm = generate_tensors(det_norm)
    det_results = det_model.detect(det_norm)
    face_boxes = det_model.get_face_boxes(det_results)

    # Check if a box is empty, if so, use previous box or next box
    for idx, box in enumerate(face_boxes):
        if len(box) == 0:
            face_boxes[idx] = face_boxes[idx - 1][:]

    # Generate deid images
    # adv_images = deid_fn.forward_batch([i.copy() for i in batch], face_boxes)
    centers, scales = align_model._get_scales_and_centers(face_boxes)
    adv_images = []
    for cv2_image, center, scale in zip(batch, centers, scales):
        _, old_box, _, _ = crop(cv2_image.copy(), center, scale, return_points=True)
        deid_image = deid_fn(cv2_image.copy(), old_box)
        adv_images.append(deid_image)

    # Stage one, attack detection
    adv_lm_imgs = attacker.attack(
        victims={"detection": det_model, "alignment": align_model},
        images=batch,
        deid_images=adv_images,
    )

    return adv_lm_imgs


def inference(images, det_model, align_model):
    # inference models on images

    # Get detection results
    results_norm = det_model.preprocess(images)
    results_norm = generate_tensors(results_norm)
    det_results = det_model.detect(results_norm)
    face_boxes = det_model.get_face_boxes(det_results)

    # Use mask to filter out empty boxes
    masks = [0 if len(box) == 0 else 1 for box in face_boxes]

    if sum(masks) == 0:
        # raise ValueError("Empty face bboxes")
        return [], []

    masked_face_boxes = [box for box, mask in zip(face_boxes, masks) if mask == 1]
    masked_images = [image for image, mask in zip(images, masks) if mask == 1]

    # Get alignment results
    centers, scales = align_model._get_scales_and_centers(masked_face_boxes)
    lm_norm = align_model.preprocess(masked_images, centers, scales)
    lm_norm = generate_tensors(lm_norm)
    _, landmarks = align_model.detect(lm_norm, centers, scales)

    # Mask empty prediction
    landmarks = [
        lm.numpy() if mask == 1 else np.zeros((68, 2))
        for lm, mask in zip(landmarks, masks)
    ]
    face_boxes = [
        box if mask == 1 else [0, 0, 0, 0] for box, mask in zip(face_boxes, masks)
    ]

    return face_boxes, landmarks


if __name__ == "__main__":

    args = parser.parse_args()

    # Init models, attackers
    if args.detector == "retinaface":
        det_model = RetinaFaceDetector()

    if args.alignment == "fan":
        align_model = FANAlignment()

    if args.deid == "pixelate":
        deid_fn = Pixelate(30)
    elif args.deid == "blur":
        deid_fn = Blur(30)

    attacker = FullAttacker(args.algorithm, n_iter=args.num_iters)

    # Read in video
    BATCH_SIZE = 2
    CAP = cv2.VideoCapture(args.video_path)
    WIDTH = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    HEIGHT = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    FPS = int(CAP.get(cv2.CAP_PROP_FPS))
    NUM_FRAMES = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT))
    VIDEO_NAME = os.path.basename(args.video_path)[:-4]
    OUTPUT_PATH = os.path.join(args.output_path, f"{VIDEO_NAME}_deid.avi")

    outvid = cv2.VideoWriter(
        OUTPUT_PATH, cv2.VideoWriter_fourcc("M", "J", "P", "G"), FPS, (WIDTH, HEIGHT)
    )

    batch = []
    frame_id = 0
    with tqdm(total=NUM_FRAMES) as pbar:
        while CAP.isOpened():
            ret, frame = CAP.read()
            if not ret:
                break

            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch.append(cv2_image)

            if (len(batch)) % BATCH_SIZE == 0 or frame_id == NUM_FRAMES - 1:

                adv_results = doit(batch, det_model, align_model, attacker, deid_fn)
                face_boxes, landmarks = inference(adv_results, det_model, align_model)
                batch = []

                if len(face_boxes) != 0:
                    for adv_img, face_box, landmark in zip(
                        adv_results, face_boxes, landmarks
                    ):
                        image = adv_img.copy()
                        plot_box(image, face_box)
                        draw_points(image, landmark)
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        outvid.write(image)
                else:
                    for adv_img in adv_results:
                        image = adv_img.copy()
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        outvid.write(image)

                pbar.update(BATCH_SIZE)

            frame_id += 1

    print(f"Attacked video is saved at {OUTPUT_PATH}")

import os
import cv2
import numpy as np
from tqdm import tqdm
from attack.attacker.full_attacker import FullAttacker
from attack.deid import Pixelate, Blur

from models import face_det, face_align, gaze_det
from models.face_align.models.face_alignment.utils import crop

from attack.attacker import generate_tensors
from utils.visualizer import *

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
parser.add_argument(
    "--gaze", "-g", type=str, default="ETH-XGaze", help="Victim gaze"
)
parser.add_argument("--algorithm", "-g", type=str, default="rmsprop", help="Algorithm")
parser.add_argument(
    "--deid", "-m", type=str, default="pixelate", help="De-identification method"
)
parser.add_argument(
    "--num_iters", type=int, default=None, help="Number of iterations to attack"
)
parser.add_argument(
    "--batch_size", "-bs", type=int, default=16, help="Batch size"
)


def deid(batch, det_model, align_model, deid_fn):

    # Generate truth bboxes
    det_norm = det_model.preprocess(batch)
    det_norm = generate_tensors(det_norm)
    det_results = det_model.detect(det_norm)
    face_boxes = det_model.get_face_boxes(det_results)

    # Check if a box is empty, if so, use previous box or next box
    for idx, box in enumerate(face_boxes):
        if len(box) == 0:
           face_boxes[idx] = face_boxes[idx-1][:]

    # Generate deid images
    centers, scales = align_model._get_scales_and_centers(face_boxes)
    adv_images = []
    for cv2_image, center, scale in zip(batch, centers, scales):
        _, old_box, _, _ = crop(cv2_image.copy(), center, scale, return_points=True)
        deid_image = deid_fn(cv2_image.copy(), old_box)
        adv_images.append(deid_image)

    return adv_images


def attack(batch, adv_images, attacker, victims):

    adv_imgs = attacker.attack(
        victims = victims, 
        images = batch, 
        deid_images = adv_images)
    
    return adv_imgs

def inference(images, det_model, align_model=None, gaze_model=None):
    # inference models on images

    # Get detection results
    results_norm = det_model.preprocess(images)
    results_norm = generate_tensors(results_norm)
    det_results = det_model.detect(results_norm)
    face_boxes = det_model.get_face_boxes(det_results)

    # Use mask to filter out empty boxes 
    masks = [0 if len(box) == 0 else 1 for box in face_boxes]
    masked_face_boxes = [box for box, mask in zip(face_boxes, masks) if mask == 1]
    masked_images = [image for image, mask in zip(images, masks) if mask == 1]

    face_boxes = [box if mask == 1 else [0,0,0,0] for box, mask in zip(face_boxes, masks)]

    if align_model is None:
        return face_boxes

    # Get alignment results
    centers, scales = align_model._get_scales_and_centers(masked_face_boxes)
    lm_norm = align_model.preprocess(masked_images, centers, scales)
    lm_norm = generate_tensors(lm_norm)
    _, masked_landmarks = align_model.detect(lm_norm, centers, scales)

    masked_landmarks = [lm.numpy() for lm in masked_landmarks]
    # Mask empty prediction
    landmarks = [lm if mask == 1 else np.zeros((68,2)) for lm, mask in zip(masked_landmarks, masks)]

    if gaze_model is None:
        return face_boxes, landmarks

    gaze_norm, faces = gaze_model.preprocess(images, masked_face_boxes, masked_landmarks, return_faces=True)
    gaze_results = gaze_model.detect(gaze_norm)
    gaze_centers, gaze_vectors = gaze_model.get_gaze_vector(gaze_results, faces)

    return face_boxes, landmarks, gaze_centers, gaze_vectors


if __name__ == "__main__":

    args = parser.parse_args()

    det_model = face_det.get_model(args.detector)
    align_model = face_align.get_model(args.alignment)
    gaze_model = gaze_det.GazeModel(args.gaze)

    if args.deid == "pixelate":
        deid_fn = Pixelate(40)
    elif args.deid == "blur":
        deid_fn = Blur(30)

    attacker = FullAttacker(args.algorithm, n_iter=args.num_iters)

    # Read in video
    BATCH_SIZE = args.batch_size
    CAP = cv2.VideoCapture(args.video_path)
    WIDTH = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    HEIGHT = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    FPS = int(CAP.get(cv2.CAP_PROP_FPS))
    NUM_FRAMES = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT))
    VIDEO_NAME = os.path.basename(args.video_path)[:-4]
    OUTPUT_PATH = os.path.join(args.output_path, f"{VIDEO_NAME}_deid.avi")
    OUTPUT_PATH2 = os.path.join(args.output_path, f"{VIDEO_NAME}_deid_viz.avi")

    outvid = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (WIDTH,HEIGHT))
    outvid2 = cv2.VideoWriter(OUTPUT_PATH2, cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (WIDTH,HEIGHT))

    batch = []
    for frame_id in tqdm(range(NUM_FRAMES)):
        if os.path.exists(f"/content/frames/{VIDEO_NAME}/{frame_id}.jpg"):
            input_img = cv2.imread(f"/content/frames/{VIDEO_NAME}/{frame_id}.jpg")
            cv2_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            batch.append(cv2_image)

        if (len(batch)+1) % BATCH_SIZE == 0 or frame_id==NUM_FRAMES-1:

            deid_images = deid(batch, det_model, align_model, deid_fn)
            adv_images = attack(batch, deid_images, attacker, 
                                victims = {
                                    'detection': det_model,
                                    'alignment': align_model,
                                    'gaze': gaze_model
                                })
            bboxes, landmarks, gaze_centers, gaze_vectors = inference(adv_images, det_model, align_model, gaze_model)

            batch = []

            if len(bboxes) != 0 :
                for adv_img, face_box, landmark, gaze_center, gaze_vector in zip(adv_images, bboxes, landmarks, gaze_centers, gaze_vectors):
                    image = adv_img.copy()
                    plot_box(image, face_box)
                    draw_points(image, landmark)

                    draw_3d_line(
                        image,
                        gaze_center, 
                        gaze_center +  0.05 * gaze_vector,
                        camera = gaze_model._face3d.camera
                    )
                    
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    outvid.write(image)
            else:
                for adv_img in adv_images:
                    image = adv_img.copy()
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    outvid.write(image)

            for adv_img in adv_images:
                image = adv_img.copy()
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                outvid2.write(image)

    print(f"Attacked video is saved at {OUTPUT_PATH}")

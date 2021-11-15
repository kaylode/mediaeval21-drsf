def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

import os
import cv2
import numpy as np
from tqdm import tqdm
from attack.attacker.full_attacker import FullAttacker
from attack.deid import Pixelate, Blur

from models import face_det, face_align, gaze_det
from models.face_align.models.face_alignment.utils import crop

from attack.attacker import generate_tensors

import argparse


parser = argparse.ArgumentParser("Video De-identification")
parser.add_argument("--input_path", "-i", type=str, required=True, help="Video input path")
parser.add_argument("--output_path", "-o", type=str, help="Video output path")
parser.add_argument("--detector", "-d", type=str, default="retinaface", help="Victim detector")
parser.add_argument("--alignment", "-a", type=str, default="fan", help="Victim alignment")
parser.add_argument("--gaze", "-z", type=str, default="ETH-XGaze", help="Victim gaze")
parser.add_argument("--algorithm", "-g", type=str, default="rmsprop", help="Algorithm")
parser.add_argument("--deid", "-m", type=str, default="pixelate_30", help="De-identification method")
parser.add_argument("--max_iter", type=int, default=150, help="Maximum number of iterations to attack")
parser.add_argument("--batch_size", "-bs", type=int, default=16, help="Batch size")

def get_width_height(path):
    extension = os.path.splitext(path)[1]
    if extension in ['.jpg', '.jpeg', '.png']:
        image = cv2.imread(path)
        h, w = image.shape[:2]
    elif extension in ['.mp4', '.avi']:
        cap = cv2.VideoCapture(path)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
    return w, h

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

def attack_video(args, det_model, align_model, gaze_model, attacker, deid_fn):
    # Read in video
    BATCH_SIZE = args.batch_size
    CAP = cv2.VideoCapture(args.input_path)
    WIDTH = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    HEIGHT = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    FPS = int(CAP.get(cv2.CAP_PROP_FPS))
    NUM_FRAMES = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT))
    VIDEO_NAME = os.path.basename(args.input_path)[:-4]
    OUTPUT_PATH = os.path.join(args.output_path, f"{VIDEO_NAME}.avi")

    outvid = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (WIDTH,HEIGHT))

    batch = []
    frame_id = 0
    with tqdm(total=NUM_FRAMES) as pbar:
        while CAP.isOpened():
            ret, frame = CAP.read()
            if not ret:
                break

            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch.append(cv2_image)

            if (len(batch)) % BATCH_SIZE == 0 or frame_id==NUM_FRAMES-1:

                deid_images = deid(batch, det_model, align_model, deid_fn)
                adv_images = attack(batch, deid_images, attacker, 
                                    victims = {
                                        'detection': det_model,
                                        'alignment': align_model,
                                        'gaze': gaze_model
                                    })
                batch = []

                for adv_img in adv_images:
                    image = adv_img.copy()
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    outvid.write(image)
                
                pbar.update(BATCH_SIZE)
            frame_id += 1
    print(f"Attacked video is saved at {OUTPUT_PATH}")

def attack_image(args, det_model, align_model, gaze_model, attacker, deid_fn):
    image = cv2.imread(args.input_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    deid_images = deid([image], det_model, align_model, deid_fn)
    adv_images = attack(
        [image], deid_images, attacker, 
        victims = {
            'detection': det_model,
            'alignment': align_model,
            'gaze': gaze_model
    })

    IMAGE_NAME = os.path.basename(args.input_path)[:-4]
    OUTPUT_PATH = os.path.join(args.output_path, f"{IMAGE_NAME}.png")
    adv_image = cv2.cvtColor(adv_images[0], cv2.COLOR_RGB2BGR)
    cv2.imwrite(OUTPUT_PATH, adv_image)
    print(f"Attacked image is saved at {OUTPUT_PATH}")
    

if __name__ == "__main__":

    args = parser.parse_args()

    width, height = get_width_height(args.input_path)
    det_model = face_det.get_model(args.detector)
    align_model = face_align.get_model(args.alignment)
    gaze_model = gaze_det.GazeModel(args.gaze, width, height)

    deid_level = args.deid.split('_')[1]
    if args.deid.startswith("pixelate"):
        deid_fn = Pixelate(int(deid_level))
    elif args.deid.startswith("blur"):
        deid_fn = Blur(int(deid_level))

    attacker = FullAttacker(args.algorithm, max_iter=args.max_iter)

    ext = os.path.splitext(args.input_path)[1]

    if ext in ['.png', '.jpeg', '.jpg']:
        attack_image(args, det_model, align_model, gaze_model, attacker, deid_fn)
    elif ext in  ['.mp4', '.avi']:
        attack_video(args, det_model, align_model, gaze_model, attacker, deid_fn)
    else:
        print("File extension is not supported")

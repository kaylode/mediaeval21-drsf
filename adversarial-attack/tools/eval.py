# write opencv read 2 videos and compare them
from tqdm.auto import tqdm
import cv2
import os
import sys
import torch
import argparse
from models import face_det, face_align, gaze_det
from attack.attacker import generate_tensors
from sklearn.metrics.pairwise import paired_euclidean_distances, paired_cosine_distances
from models.gaze_det.ptgaze.utils import compute_angle_error

parser = argparse.ArgumentParser("Video De-identification Evaluation")
parser.add_argument("video1", type=str, help="Video path")
parser.add_argument("video2", type=str, help="Video path")
parser.add_argument("--detector", "-d", type=str, default="retinaface", help="Victim detector")
parser.add_argument("--alignment", "-a", type=str, default="fan", help="Victim alignment")
parser.add_argument("--gaze", "-z", type=str, default="ETH-XGaze", help="Victim gaze")

def calc_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


class Evaluator:
    def __init__(self, det_model, align_model=None, gaze_model=None):
        self.det_model = det_model
        self.align_model = align_model
        self.gaze_model = gaze_model

    def _evaluate_detection(self, frame1, frame2):
        results_norm = self.det_model.preprocess([frame1, frame2])
        results_norm = generate_tensors(results_norm)
        det_results = self.det_model.detect(results_norm)
        face_boxes = self.det_model.get_face_boxes(det_results)

        iou_score = calc_iou(face_boxes[0], face_boxes[1])
        return iou_score, face_boxes

    def _evaluate_alignment(self, frame1, frame2, bboxes):
        centers, scales = self.align_model._get_scales_and_centers(bboxes)
        lm_norm = self.align_model.preprocess([frame1, frame2], centers, scales)
        lm_norm = generate_tensors(lm_norm)
        _, landmarks = self.align_model.detect(lm_norm, centers, scales)

        landmarks = [lm.numpy() for lm in landmarks]

        dist = paired_euclidean_distances(landmarks[0], landmarks[1]).mean()
        return dist, landmarks

    def _evaluate_gaze(self, frame1, frame2, bboxes, landmarks):
        gaze_norm, faces = self.gaze_model.preprocess([frame1, frame2], bboxes, landmarks, return_faces=True)
        gaze_results = self.gaze_model.detect(gaze_norm)
        
        for face, gaze_vector in zip(faces, gaze_results):
            self.gaze_model._face3d.postprocess([gaze_vector], face)

        euler_angles1 = faces[0].gaze_vector[:2].reshape((1, 2))
        euler_angles2 = faces[1].gaze_vector[:2].reshape((1, 2))
        gaze_dist = paired_euclidean_distances(euler_angles1, euler_angles2).mean() 
        angle_error = compute_angle_error(torch.from_numpy(gaze_results[0]).unsqueeze(0), torch.from_numpy(gaze_results[1]).unsqueeze(0))
        angle_error = float(angle_error.item())
        return gaze_dist, angle_error

    def evaluate(self, frame1, frame2):

        eval_results = {}
        iou_score, bboxes = self._evaluate_detection(frame1, frame2)
        eval_results['box_iou'] = iou_score

        if self.align_model is not None:
            lm_dist, landmarks = self._evaluate_alignment(frame1, frame2, bboxes)
            eval_results['lm_edist'] = lm_dist

        if self.gaze_model is not None:
            gaze_dist, angle_error = self._evaluate_gaze(frame1, frame2, bboxes, landmarks)
            eval_results['gaze_edist'] = gaze_dist
            eval_results['angle_error'] = angle_error
            eval_results['diffscore'] = (1-iou_score) + lm_dist + angle_error
        return eval_results

class AvgMeter:
    def __init__(self):
        super().__init__()
        self.reset()

    def update(self, item):
        for key, value in item.items():
            if key not in self.data:
                self.data[key] = 0
            self.data[key] += value

        self.sample_size += 1

    def value(self):
        for keys in self.data:
            self.result[keys] = self.data[keys] / self.sample_size
        return self.result

    def reset(self):
        self.data = {}
        self.result = {}
        self.sample_size = 0

    def summary(self):
        results = self.value()
        print(f"Evaluate {self.sample_size} frames")
        for key, value in results.items():
            print(f"{key} : {value:.6f}")


if __name__ == "__main__":
    args = parser.parse_args()
    # get video file path
    video_path = args.video1
    video_path2 = args.video2
    # get video file name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_name2 = os.path.splitext(os.path.basename(video_path2))[0]
    # get video object
    cap = cv2.VideoCapture(video_path)
    cap2 = cv2.VideoCapture(video_path2)
    # get video fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    # get video width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # get video length
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    # get video duration
    duration = length / fps
    duration2 = length2 / fps2
    # check if video is open
    if not cap.isOpened() or not cap2.isOpened():
        print("Error opening video stream or file")
        sys.exit(1)
    meter = AvgMeter()
    total = min(length, length2)
    pbar = tqdm(total=total)
    # write tqdm progress bar for while

    det_model = face_det.get_model(args.detector)
    align_model = face_align.get_model(args.alignment)
    gaze_model = gaze_det.GazeModel(args.gaze, width, height)
    evaluator = Evaluator(det_model, align_model, gaze_model)

    while True:
        # get current frame
        ret, frame = cap.read()
        ret2, frame2 = cap2.read()
        # check if frame is read
        if not ret or not ret2:
            break
        try:
            results = evaluator.evaluate(frame, frame2)
        except:
            pbar.update(1)
            continue
        meter.update(results)
        pbar.update(1)

    # release video object
    if ret == True:
        cap.release()

    if ret2 == True:
        cap2.release()

    cv2.destroyAllWindows()
    meter.summary()

# write opencv read 2 videos and compare them
from tqdm.auto import tqdm
import cv2
import os
import json
import argparse
from models import face_det, face_align, gaze_det
from attack.attacker import generate_tensors


parser = argparse.ArgumentParser("Video De-identification Evaluation")
parser.add_argument("--input_path", '-i', type=str, help="Video path")
parser.add_argument("--output_path", '-o', type=str, help="Output JSON path")
parser.add_argument("--detector", "-d", type=str, default="retinaface", help="Victim detector")
parser.add_argument("--alignment", "-a", type=str, default="fan", help="Victim alignment")
parser.add_argument("--gaze", "-z", type=str, default="ETH-XGaze", help="Victim gaze")


class Predictor:
    def __init__(self, det_model, align_model=None, gaze_model=None):
        self.det_model = det_model
        self.align_model = align_model
        self.gaze_model = gaze_model

    def _predict_detection(self, frame):
        results_norm = self.det_model.preprocess([frame])
        results_norm = generate_tensors(results_norm)
        det_results = self.det_model.detect(results_norm)
        face_boxes = self.det_model.get_face_boxes(det_results)

        return face_boxes

    def _predict_alignment(self, frame, bboxes):
        centers, scales = self.align_model._get_scales_and_centers(bboxes)
        lm_norm = self.align_model.preprocess([frame], centers, scales)
        lm_norm = generate_tensors(lm_norm)
        _, landmarks = self.align_model.detect(lm_norm, centers, scales)

        landmarks = [lm.numpy() for lm in landmarks]

        return landmarks

    def _predict_gaze(self, frame, bboxes, landmarks):
        gaze_norm, faces = self.gaze_model.preprocess([frame], bboxes, landmarks, return_faces=True)
        gaze_results = self.gaze_model.detect(gaze_norm)
        
        for face, gaze_vector in zip(faces, gaze_results):
            self.gaze_model._face3d.postprocess([gaze_vector], face)

        gaze_vector = faces[0].gaze_vector[:2].reshape((1, 2)) #.head_pose_rot.as_euler("XYZ", degrees=True).reshape((1, 3))
        return gaze_results, gaze_vector

    def predict(self, frame):

        predict_results = {}
        bboxes = self._predict_detection(frame)
        predict_results['bboxes'] = bboxes[0]

        landmarks = self._predict_alignment(frame, bboxes)
        predict_results['landmarks'] = [i.tolist() for i in landmarks][0]

        gaze_results, gaze_vector = self._predict_gaze(frame, bboxes, landmarks)
        predict_results['gaze_angle'] = gaze_results.tolist()[0]
        predict_results['gaze_vector'] = gaze_vector.tolist()[0]
        return predict_results


if __name__ == "__main__":
    args = parser.parse_args()
    # get video file path
    video_path = args.input_path
    # get video file name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # get video object
    cap = cv2.VideoCapture(video_path)
    # get video fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    # get video width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # get video length
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get video duration
    duration = length / fps

    pbar = tqdm(total=length)
    # write tqdm progress bar for while

    det_model = face_det.get_model(args.detector)
    align_model = face_align.get_model(args.alignment)
    gaze_model = gaze_det.GazeModel(args.gaze, width, height)
    evaluator = Predictor(det_model, align_model, gaze_model)

    result_list = []

    frame_idx = 0
    while True:
        result_dict = {}
        result_dict['frame_id'] = frame_idx
        ret, frame = cap.read()
        if not ret:
            break
        results = evaluator.predict(frame)
        result_dict.update(results)
        # result_dict['bboxes'] = []
        # result_dict['landmarks'] = []
        # result_dict['gaze_vector'] = []
        # result_dict['euler_angles'] = []

        frame_idx += 1
        pbar.update(1)

        result_list.append(result_dict)

    # release video object
    if ret == True:
        cap.release()

    cv2.destroyAllWindows()

    with open(args.output_path, 'w') as f:
        json.dump(result_list, f)

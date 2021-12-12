# write opencv read 2 videos and compare them
from tqdm.auto import tqdm
import json
import torch
import argparse
from sklearn.metrics.pairwise import paired_euclidean_distances, paired_cosine_distances
from models.gaze_det.ptgaze.utils import compute_angle_error

parser = argparse.ArgumentParser("Video De-identification Evaluation")
parser.add_argument("--json_gt", type=str, help="Json groundtruth")
parser.add_argument("--json_pred", type=str, help="JSON prediction")

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

class Evaluator:
    def __init__(self, gt_json_path, pred_json_path):
        with open(gt_json_path, 'r') as f:
            self.gt_json = json.load(f)

        with open(pred_json_path, 'r') as f:
            self.pred_json = json.load(f)

        self.sample_size = len(self.gt_json)

    def _evaluate_detection(self, face_boxes1, face_boxes2):
        iou_score = calc_iou(face_boxes1, face_boxes2)
        return iou_score

    def _evaluate_alignment(self, landmarks1, landmarks2):
        dist = paired_euclidean_distances(landmarks1, landmarks2).mean()
        return dist

    def _evaluate_gaze(self, gaze1, gaze2, gaze_vector1, gaze_vector2):
        gaze_edist = paired_euclidean_distances([gaze_vector1], [gaze_vector2]).mean()

        angle_error = compute_angle_error(torch.FloatTensor([gaze1]), torch.FloatTensor([gaze2]))
        angle_error = float(angle_error.item())
        return gaze_edist, angle_error

    def evaluate(self, dict1, dict2):

        eval_results = {}
        iou_score = self._evaluate_detection(dict1['bboxes'], dict2['bboxes'])
        eval_results['box_iou'] = iou_score

        lm_dist = self._evaluate_alignment(dict1['landmarks'], dict2['landmarks'])
        eval_results['lm_edist'] = lm_dist

        gaze_dist, angle_error = self._evaluate_gaze(
            dict1['gaze_angle'], 
            dict2['gaze_angle'],
            dict1['gaze_vector'], 
            dict2['gaze_vector']
        )
        eval_results['gaze_euclide_dist'] = gaze_dist
        eval_results['angle_error'] = angle_error
        eval_results['diffscore'] = (1-iou_score) + lm_dist + gaze_dist
        return eval_results

    def get_results(self):
        meter = AvgMeter()
        pbar = tqdm(total=self.sample_size)
        for dict1, dict2 in zip(self.pred_json, self.gt_json):
            frame_id1 = dict1['frame_id']
            frame_id2 = dict1['frame_id']
            if frame_id1 != frame_id2:
                print("Missing frame. End evaluation here")
                break

            results = self.evaluate(dict1, dict2)
            meter.update(results)
            pbar.update(1)

        meter.summary()

if __name__ == "__main__":
    args = parser.parse_args()
    
    evaluator = Evaluator(args.json_pred, args.json_gt)
    evaluator.get_results()
# Driving Road Safety Forward: Video Data Privacy

## Example use cases

### Non-targeted attack

- Attack on batch

```python
from attack.deid import Pixelate
from models.face_align.fan import FANAlignment
from models.face_det.retinaface import RetinaFaceDetector
from attack.attacker import FullAttacker, generate_tensors
from models.gaze_det.gaze import GazeModel

# Init models, attackers
align_model = FANAlignment()
det_model = RetinaFaceDetector()
gaze_model = GazeModel('ETH-XGaze')
attacker = FullAttacker('rmsprop')
deid_fn = Pixelate(40)

def doit(batch):
    """
    Attack batch of images on detection and alignment models
    :params:
        batch: list of cv2 image
    :return: list of adversarial images
    """
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

    deid_images = []
    for cv2_image, center, scale in zip(batch, centers, scales):
        _, old_box, _, _ = crop(cv2_image.copy(), center, scale, return_points=True)
        deid_image = deid_fn(cv2_image.copy(), old_box)
        deid_images.append(deid_image)

    # Stage one, attack detection
    adv_images = attacker.attack(
        victims = {
            'detection': det_model,
            'alignment': align_model,
            'gaze': gaze_model
        },
        images = batch,
        deid_images = deid_images)

    return adv_images
```

|                       Original images                        |                   Predictions before deid                    |                    Predictions after deid                     |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :-----------------------------------------------------------: |
| <img width="450" alt="screen" src="assets/results/ori.jpg">  | <img width="450" alt="screen" src="assets/results/raw.jpg">  | <img width="450" alt="screen" src="assets/results/deid.jpg">  |
| <img width="450" alt="screen" src="assets/results/ori2.jpg"> | <img width="450" alt="screen" src="assets/results/raw2.jpg"> | <img width="450" alt="screen" src="assets/results/deid2.jpg"> |
| <img width="450" alt="screen" src="assets/results/ori3.jpg"> | <img width="450" alt="screen" src="assets/results/raw3.jpg"> | <img width="450" alt="screen" src="assets/results/deid3.jpg"> |
| <img width="450" alt="screen" src="assets/results/ori4.jpg"> | <img width="450" alt="screen" src="assets/results/raw4.jpg"> | <img width="450" alt="screen" src="assets/results/deid4.jpg"> |
| <img width="450" alt="screen" src="assets/results/ori5.jpg"> | <img width="450" alt="screen" src="assets/results/raw5.jpg"> | <img width="450" alt="screen" src="assets/results/deid5.jpg"> |


## Evaluation

```bash
python evaluation.py    <video1> <video2> \
                        -d [retinaface, mtcnn] \
                        -a [fan] \
                        -g [MPIIFaceGaze, ETH-XGaze]
```

## Colab Notebooks

- Adversarial Attack [![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BXiBrxdfAK2JEW2uU7ZshKLPbD4ZSXXb?usp=sharing)

## Code References

- https://github.com/biubug6/Pytorch_Retinaface
- https://github.com/timesler/facenet-pytorch
- https://github.com/1adrianb/face-alignment
- https://github.com/hysts/pytorch_mpiigaze
- https://github.com/git-disl/TOG
- https://github.com/honguyenhaituan/PrivacyPreservingFaceRecognition

## Paper References

- https://github.com/brighter-ai/awesome-privacy-papers

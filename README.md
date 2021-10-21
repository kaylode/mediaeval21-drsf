# Driving Road Safety Forward: Video Data Privacy

## Example use case
### Non-targeted attack
```python
from models.retinaface import MTCNNDetector
from attack.attacker import Attacker
from attack.deid import Pixelate

input_img = cv2.imread("./assets/driver_face.JPG")
cv2_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

attacker = Attacker(optim='RMSprop')   # Use RMSprop method
x_adv = attacker.attack(
    cv2_image = cv2_image,            # query image
    detector = MTCNNDetector(),       # attack mtcnn
    deid_fn = Pixelate(10),           # use pixelate method
    min_value = -1)                   # mtcnn requires
    
plt.imshow(x_adv)
```

| Input image | Model prediction after deid + attack |
|:-------------------------:|:-------------------------:|
|<img width="450" alt="screen" src="assets/test_images/paul_rudd/1.jpg"> | <img width="450" alt="screen" src="assets/deid2.jpg"> |

### Targeted attack
```python
from models.retinaface import RetinaFaceDetector
from attack.attacker import Attacker
from attack.deid import Pixelate

input_img = cv2.imread("./assets/driver_face.JPG")
cv2_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

targets = [tensor([[182.4111,  64.1539, 309.9272, 243.2287, 216.0245, 131.3289, 276.3042,
          132.5408, 244.9431, 167.0494, 221.2922, 198.1093, 269.2490, 198.9554,
            0.9995]])]
face_box = [182, 64, 309, 243]

attacker = Attacker(optim='I-FGSM')   # Use IFGSM method
x_adv = attacker.attack(
    cv2_image = cv2_image,            # query image
    detector = RetinaFaceDetector(),  # attack retinaface
    deid_fn = Pixelate(10),           # use pixelate method
    face_box = face_box,            
    targets = targets)
    
plt.imshow(x_adv)
```

| Model prediction | Model prediction after deid + attack |
|:-------------------------:|:-------------------------:|
|<img width="450" alt="screen" src="assets/raw.jpg"> | <img width="450" alt="screen" src="assets/deid.jpg"> |

## Colab Notebooks
- Pixelate Landmarks [![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nhtWSODf3UD7ptKLLzneAbE9MtRq-q-7?usp=sharing)
- Adversarial Attack [![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ILpV_ovjboPpmqmImZZBEf-Rmv9mRoN8?usp=sharing)

## Code References
- https://github.com/timesler/facenet-pytorch
- https://github.com/git-disl/TOG
- https://github.com/honguyenhaituan/PrivacyPreservingFaceRecognition

## Paper References

- https://github.com/brighter-ai/awesome-privacy-papers

```
@inproceedings{letournel2015face,
  title={Face de-identification with expressions preservation},
  author={Letournel, Geoffrey and Bugeau, Aur{\'e}lie and Ta, V-T and Domenger, J-P},
  booktitle={2015 IEEE International Conference on Image Processing (ICIP)},
  pages={4366--4370},
  year={2015},
  organization={IEEE}
}
```

# Submission description

- Our submission contains 2 runs:

    - **HCMUS_Team_faceswap.zip**: 
        - In this run, we detect the face from input image and video. Then using a network to map the face from image into the target video.

    - **HCMUS_Team_adversarial.zip**: 
        - Folder ```submission``` contains 720 de-identified videos, these videos have been attacked by our algorithms which all the face of the drivers are pixelated while still preserving its deep features when inputed into ```victim models```

        - Folder ```visualization``` contains 720 videos that are the same as above folder. But they are forwarded again through the ```victim models``` to predict face bounding box, facial landmarks, gaze vector, then the results are drawn on these videos. Also in each action category folder, there are ```config.txt``` file which describe the configuration we apply to attack the entire folder.

        - Folder ```source code``` contains code for this proposed run. This folder also has evaluation scripts that we want to propose to judge our methods. Please read the README.md for instruction to run.
        

* Some keywords:
    - ```victim models```: all models which are the victims of our adversarial attack methods. Which are RetinaFace, FAN Alignment and MPIIFaceGaze, ETH-XGaze
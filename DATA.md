## Outline


## General Overview of Data

###  What is it?
The SHRP2 dataset collected as part of the Naturalistic Driving Study (NDS) contains millions of hours and petabytes of driver video data that can be used by researchers to gain a better understanding of what causes car crashes. This dataset is currently hosted in a secure enclave, making it inaccessible to most of the research community due to the personal nature of video and lack of consent from the drivers to have the videos released publicly. 

The dataset consists of both high- and low-resolution driver video  data prepared by Oak Ridge National Laboratory for this Driver Video  Privacy Task. The data were captured using the same data acquisition  system as the larger SHRP2 dataset mentioned above, which currently has  limited access in a secure enclave. For the data in this Task, there are  drivers in choreographed situations designed to emulate different  naturalistic driving environments. Actions include talking, coughing,  singing, dancing, waving, eating, and various others [8]. Through this  unique partnership, annotated data from Oak Ridge National Laboratory  will be available to registered participants, alongside experts from the  data collection and processing team who will be available for mentoring  and any questions.


### Levels of Difficulty
The videos are organized into folders based on estimated difficulty to perform de-identification. This difficulty is based on a combination of video quality as well as the action being performed in the video. Generally, Level 1 data should be easier to perform de-identification and action preservation than Level 2, which should also be easier than Level 3. Some individual videos in each Level may stand out as being particularly difficult but in general the overall performance for each Level will be more important than performance on one particular video.

## Download Dataset
When you signed up for the competition you were given access to a google drive containing a folder called data. Inside the data folder you will find the videos corresponding to all 3 levels of difficulty, RetinaFace detection coordinates and a script that can be used for organizing the videos by action.

## How to Use Video Parser Script

### Purpose
The provided video_parser.exe script will organize the videos according to their actions within each level. For example, running the video parser in the initial data folder will create 3 new folders: 1) Level 1 actions, 2) Level 2 actions, 3) Level 3 actions. Each folder will contain sub-folders titled according to the actions present in the videos contained within each folder. This is included as participants may find it useful to have some of the video data organized according to actions in order to test their implementation on specific actions.

### Windows
If you have Python installed, simply open the command prompt, navigate to the appropriate folder and type the command `python video_parser.py` . Sometimes you may be required to use `python3` instead of `python` when executing this command.

Alternatively, you can navigate to the same folder and simply run the `video_parser.exe` file.

<b>Note:</b> Do not move the scripts to a different location

### Mac
Make sure you have python >= 3.5 installed and run the script `video_parser.py` inside the Videos folder.

### Installing Python Using Anaconda

It is recommended to use Anaconda to install Python on your local machine. The installation instructions for windows are [here](https://docs.anaconda.com/anaconda/install/windows/). 

Here are some helpful links to help get started with Anaconda:
- https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html


### Output
Running the python script will create the folders `Level 1_actions`, `Level 2_actions` and `Level 3_actions` which each contain subfolders titled according to a specific action. Inside each of these folders, you will find videos corresponding to the action in the video.

You can re-run the above script if you accidentally delete or corrupt some of the files to get a fresh copy of the directories containing the videos. If you delete some of the videos from the original `Level 1`, `Level 2` or `Level 3`, the script will not replace any of the original files and will simply create action folders based on the current available files.  Notice that action folders will not be created if there are no videos corresponding to a specific action in their respective difficulty groups.

## How to use RetinaFace Detection Coordinates

The RetinaFace Detection coordinates are presented in csv files for each video using a similar naming convention. The x, y, w, h coordinates correspond to a facial bounding box and the rest of the coordinates represent a 5-point landmark of the face. It is recommended to use pandas to easily work with csv files in Python.


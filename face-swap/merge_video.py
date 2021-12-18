import os 

from pathlib import Path
from tqdm import tqdm
import cv2
root_path = Path("/home/ptthang/videoprivacytask-master/")
source_path = "data"
target_path = "output_frame"
level = ['Level 1', 'Level 2', 'Level 3']
target_vid_path = 'video_cropped'
for l in level:
    sp = root_path/source_path/l 
    for f in tqdm(sp.glob("*"), total=240):
        op = root_path/target_path/l/f.name.split(".")[0] 
        op_cropped = root_path/target_vid_path/l/f.name.split(".")[0] 
        os.makedirs(root_path/target_vid_path/l, exist_ok=True)
        vid_path = str(f)
        vidcap = cv2.VideoCapture(vid_path)
        fps = int(vidcap.get(cv2.CAP_PROP_FPS))
        # print(op)
        os.system('ffmpeg -y -r {} -f image2 -i "{}/frame%04d.jpg" -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -shortest -strict -2 "{}.mp4"'.format(fps,op,op_cropped))
        # break
    # break
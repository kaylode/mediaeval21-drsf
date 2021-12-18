import cv2

def get_bbox_image(img, fa):
    preds = fa.get_landmarks(img)
    kpt = preds[0].squeeze()
    left = int(np.min(kpt[:, 0]))
    right = int(np.max(kpt[:, 0]))
    top = int(np.min(kpt[:, 1]))
    bottom = int(np.max(kpt[:, 1]))
    return left, top, right, bottom


def create_logger(name='profiling',fn='profile.log', mode='a'):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(fn, mode=mode)
    fh.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s|%(name)s|%(levelname)s||%(filename)s - %(lineno)d - %(funcName)s: %(message)s\n')
    fh.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    return name,logger 

def square_bbox_from_rect_bbox(left, top, right, bottom, k_margin=0.0, shape=(256, 256)):
    height = bottom - top
    width = right - left

    more_h = 0
    more_w = 0
    if width < height:
        more_w += height - width
        width = height
    elif height < width:
        more_h = width - height
        height = width

    k = k_margin
    more_h += int(height*k)
    more_w += int(width*k)

    top = top - int(more_h * 0.5)
    more_h -= int(more_h * 0.5)
    if top < 0:
        more_h += np.abs(top)
        top = 0
    bottom = bottom + more_h
    more_h = 0
    if bottom > shape[0]:
        more_h += bottom - shape[0]
        bottom = shape[0]
    top = top - more_h
    top = max(top, 0)

    left = left - int(more_w * 0.5)
    more_w -= int(more_w * 0.5)
    if left < 0:
        more_w += np.abs(left)
        left = 0
    right += more_w
    more_w = 0
    if right > shape[1]:
        more_w += right - shape[1]
        right = shape[1]
    left -= more_w
    left = max(left, 0)
    return left, top, right, bottom


def get_all_bbox_video(vidcap, fa):
    success, image = vidcap.read()
    # left, top, right, bottom list
    bbox_list = [[], [], [], []]
    count = 0 
    while success:
        try:
            left, top, right, bottom = get_bbox_image(image, fa)
            bbox_list[0].append(left)
            bbox_list[1].append(top)
            bbox_list[2].append(right)
            bbox_list[3].append(bottom)
        except:
            logger.info("Missing face at {}".format(count))
        success, image = vidcap.read()
        count += 1
    return bbox_list


def get_final_bbox(bbox_list, shape=(256,256)):
    f_left = np.min(bbox_list[0])
    f_top = np.min(bbox_list[1])
    f_right = np.max(bbox_list[2])
    f_bottom = np.max(bbox_list[3])

    f_left, f_top, f_right, f_bottom = square_bbox_from_rect_bbox(
        f_left, f_top, f_right, f_bottom, 0.1, shape=shape)
    return int(f_left), int(f_top), int(f_right), int(f_bottom)


def crop_fixed_bbox_and_save(bbox, vidcap, tmpfolder="tmpframe"):
    os.makedirs(tmpfolder, exist_ok=True)
    shutil.rmtree(tmpfolder)
    os.makedirs(tmpfolder, exist_ok=True)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    index_frame = {}
    for i in range(0,length):
        success, image = vidcap.read()
        index_frame[i] = "miss"
        if success:
            cv2.imwrite(os.path.join(tmpfolder, "frame%04d.jpg" % i),
                        image[bbox[1]:bbox[3], bbox[0]:bbox[2]])     # save frame as JPEG file
            count += 1
            index_frame[i] = "success"
    
    with open(os.path.join(tmpfolder, "index_frames.json"), 'w') as f:
        json.dump(index_frame, f)
    with open(os.path.join(tmpfolder, "bbox_cropped.json"), 'w') as f:
        json.dump(bbox, f)

    return count 
import face_alignment
import numpy as np 
import os 
import logging 
import json
import shutil
name_logger, logger = create_logger(fn='profile-overlayvid.log',mode='w')
# import sys
# from memory_profiler import LogFile
# sys.stdout = LogFile('profiling', reportIncrementFlag=False)

from pathlib import Path
from tqdm import tqdm
import pandas as pd 
import numpy as np 

root_path = Path("/home/ptthang/videoprivacytask-master/")
source_path = "data"
target_path = "video_cropped_output_newface"
annotation = "data/RetinaFaceDetections"
level = ['Level 1', 'Level 2', 'Level 3']
for l in level:
    sp = root_path/source_path/l 
    for f in tqdm(sp.glob("*"), total=240):
        # op = root_path/target_path/l/f.name.split(".")[0] 
        # os.makedirs(op, exist_ok=True)
        fname = f.name.split(".")[0] +'.csv'
        anno_fn = root_path/annotation/l/fname
        anno = pd.read_csv(anno_fn)
        x_col = np.array(anno['x'].values)
        y_col = np.array(anno['y'].values)
        w_col = np.array(anno['w'].values)
        h_col = np.array(anno['h'].values)
        x2_col = x_col + w_col
        y2_col = y_col + h_col
        bbox_list = [x_col,y_col,x2_col, y2_col]
        vid_path = str(f)
        vidcap = cv2.VideoCapture(vid_path)
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # bbox_list =get_all_bbox_video(vidcap,fa)
        height, width = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        f_left, f_top, f_right, f_bottom  = get_final_bbox(bbox_list,shape=(height, width))

        # source vid path
        # if f.name != 'T008_ActionsShorter_Face_12900_13187_Dance-Sing.mp4':
        #     continue
        fps_vid = vidcap.get(cv2.CAP_PROP_FPS)
        # print(vidcap.get(cv2.CAP_PROP_FPS))
        output_vid_resized = root_path/target_path/l/f.name.split(".")[0]/'result_resized_opriginal.avi'
        output_vid_path = root_path/target_path/l/f.name.split(".")[0]/'result.mp4'
        output_vid_overlay = root_path/target_path/l/f.name.split(".")[0]/'result-overlay.mp4'
        if output_vid_path.exists() == False:
            print('{} not found!'.format(output_vid_path))
            break
        if f.exists() == False:
            print('{} not found!'.format(f))
            break
        if output_vid_resized.exists() == True:
            os.remove(output_vid_resized)
        os.system('ffmpeg -y -r {} -i "{}" -s {}x{} -c:a copy "{}" 2>> /home/ptthang/videoprivacytask-master/profile-overlayvid.log'.format(fps_vid,output_vid_path, f_right-f_left+1, f_bottom-f_top+1,output_vid_resized))
        if output_vid_resized.exists() == False:
            print('{} not found!'.format(output_vid_resized))
            break
        if output_vid_overlay.exists() == True:
            os.remove(output_vid_overlay)
        os.system('ffmpeg -y -i "{}" -i "{}" -filter_complex "[0:v][1:v] overlay={}:{}" -pix_fmt yuv420p -c:a copy "{}" 2>> /home/ptthang/videoprivacytask-master/profile-overlayvid.log'.format(vid_path, output_vid_resized,f_left+1, f_top+1, output_vid_overlay))
        # videoprivacytask-master/video_cropped_output/Level 1/T002_ActionsShorter_mini_3239_3347_Use-Radio-or-Gadget/result.mp4



        # vidcap = cv2.VideoCapture(vid_path)
        # success = crop_fixed_bbox_and_save([f_left, f_top, f_right, f_bottom],vidcap, op)

        # logger.info("fn: '{}', frames: {}, read sucess: {}, bbox: {}".format(f, length, success, (f_left, f_top, f_right, f_bottom)))

    #     break
    # break
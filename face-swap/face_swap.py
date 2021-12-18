from face_swap_modules.part_swap import load_checkpoints, make_video, load_face_parser

from skimage.transform import resize
from skimage import img_as_ubyte
import imageio
import moviepy.editor as me

from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import os

def concat_frame(inp1, inp2, inp3):
    out = np.concatenate((inp1, inp2), axis=1)
    out = np.concatenate((out, inp3), axis=1)
    return out

def add_audio(output_path, driving_path):
    audio = me.VideoFileClip(driving_path).audio

    video = me.VideoFileClip(output_path)
    video = video.set_audio(audio)
    
    out_path = Path(output_path)
    parent_path = out_path.parent
    name_path = 'ad-' + out_path.name
    out_path = str(parent_path.joinpath(name_path))

    video.write_videofile(out_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='config/vox-256-sem-10segments.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='checkpoints/vox-first-order.pth.tar', help="path to checkpoint to restore")
    parser.add_argument("--source_image", default='example_face.png', help="path to src image")
    parser.add_argument("--target_video", default='video/2.mp4', help="path to target video")
    parser.add_argument("--result_video", default='demo/save_swap', help="path to save")
    parser.add_argument("--mode", default='face', help="choose in ['face', 'hair', 'all']")

    parser.add_argument("--blend_scale", default=0.125, type=float)

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    opt = parser.parse_args()

    reconstruction_module, segmentation_module = load_checkpoints(config=opt.config, \
                                               checkpoint=opt.checkpoint, \
                                               blend_scale=opt.blend_scale, first_order_motion_model=True)
    
    source_image = imageio.imread(opt.source_image)
    reader = imageio.get_reader(opt.target_video)
    fps = reader.get_meta_data()['fps']
    target_video = []
    try:
        for im in reader:
            target_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    source_image = resize(source_image, (256, 256))[..., :3]
    target_video = [resize(frame, (256, 256))[..., :3] for frame in target_video]
    

    if opt.cpu:
        face_parser = load_face_parser(cpu=True)
    else:
        face_parser = load_face_parser(cpu=False)

    if opt.mode == 'face':
        swap_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    elif opt.mode == 'hair':
        swap_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17]
    else:
        swap_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]


    predictions = make_video(swap_index = swap_index, source_image = source_image,\
                         target_video = target_video, use_source_segmentation=True, segmentation_module=segmentation_module, \
                         reconstruction_module=reconstruction_module, face_parser=face_parser)

    output = [concat_frame(source_image, predictions[i], target_video[i]) for i in range(len(predictions))]

    imageio.mimsave(os.path.join(opt.result_video, 'output.mp4'), [img_as_ubyte(frame) for frame in output], fps=fps)
    imageio.mimsave(os.path.join(opt.result_video, 'result.mp4'), [img_as_ubyte(frame) for frame in predictions], fps=fps)
    
    add_audio(os.path.join(opt.result_video, 'result.mp4'), opt.target_video)
    add_audio(os.path.join(opt.result_video, 'output.mp4'), opt.target_video)




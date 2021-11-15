from omegaconf import DictConfig

from models.gaze_det.ptgaze.utils import (
    check_path_all,
    generate_dummy_camera_params,
)

def get_config(name):

    if name == 'ETH-XGaze':
        test_params = {
            "mode": "ETH-XGaze",
            "device": "cpu",
            "model": {"name": "resnet18"},
            "gaze_estimator": {
                "checkpoint": "./assets/pretrained/eth-xgaze_resnet18.pth",
                "use_dummy_camera_params": True,
                "normalized_camera_params": "./models/gaze_det/ptgaze/configs/normalized_camera_params/eth-xgaze.yaml",
                "normalized_camera_distance": 0.6,
                "image_size": [224, 224],
            },
            "demo": {
                "use_camera": False,
                "display_on_screen": False,
                "wait_time": 1,
                "image_path": None,
                "video_path": "./assets/T002_ActionsShorter_mini_8829_9061_Talk-non-cell.mp4",
                "output_dir": ".",
                "output_file_extension": "avi",
                "head_pose_axis_length": 0.05,
                "gaze_visualization_length": 0.05,
                "show_bbox": True,
                "show_head_pose": True,
                "show_landmarks": False,
                "show_normalized_image": False,
                "show_template_model": True,
            },
        }

    elif name == 'MPIIFaceGaze':
        test_params = {
            "mode": "MPIIFaceGaze",
            "device": "cpu",
            "model": {"name": "resnet_simple", "backbone": {
                "name": "resnet_simple",
                "pretrained": "resnet18",
                "resnet_block": "basic",
                "resnet_layers": [2, 2, 2]
            }},
            "gaze_estimator": {
                "checkpoint": "./assets/pretrained/mpiifacegaze_resnet_simple.pth",
                "use_dummy_camera_params": True,
                "normalized_camera_params": "./models/gaze_det/ptgaze/configs/normalized_camera_params/mpiifacegaze.yaml",
                "normalized_camera_distance": 0.6,
                "image_size": [224, 224],
            },
            "demo": {
                "use_camera": False,
                "display_on_screen": False,
                "wait_time": 1,
                "image_path": None,
                "video_path": "./assets/T002_ActionsShorter_mini_8829_9061_Talk-non-cell.mp4",
                "output_dir": ".",
                "output_file_extension": "avi",
                "head_pose_axis_length": 0.05,
                "gaze_visualization_length": 0.05,
                "show_bbox": True,
                "show_head_pose": True,
                "show_landmarks": False,
                "show_normalized_image": False,
                "show_template_model": True,
            },
        }
    else:
        raise Exception("Only ETH-XGaze and MPIIFaceGaze are supported")

    config = DictConfig(test_params)
    if config.gaze_estimator.use_dummy_camera_params:
        generate_dummy_camera_params(config)
    return config
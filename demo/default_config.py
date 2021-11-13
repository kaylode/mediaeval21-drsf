from omegaconf import DictConfig

test_params = {
    "mode": "ETH-XGaze",
    "device": "cpu",
    "model": {"name": "resnet18"},
    "gaze_estimator": {
        "checkpoint": "./demo/pretrained/eth-xgaze_resnet18.pth",
        "use_dummy_camera_params": True,
        "normalized_camera_params": "./demo/data/normalized_camera_params/eth-xgaze.yaml",
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
    "PACKAGE_ROOT": "./",
}
config = DictConfig(test_params)

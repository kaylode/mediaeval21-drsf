from omegaconf import DictConfig

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
        }

    elif name == 'MPIIFaceGaze':
        test_params = {
            "mode": "MPIIFaceGaze",
            "device": "cpu",
            "model": {"name": "resnet_simple"},
            "gaze_estimator": {
                "checkpoint": "./assets/pretrained/mpiifacegaze_resnet_simple.pth",
                "use_dummy_camera_params": True,
                "normalized_camera_params": "./models/gaze_det/ptgaze/configs/normalized_camera_params/mpiifacegaze.yaml",
                "normalized_camera_distance": 0.6,
                "image_size": [224, 224],
            },
        }

    config = DictConfig(test_params)
    return config

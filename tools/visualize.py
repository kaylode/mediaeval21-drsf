import datetime
import logging
import pathlib
from typing import Optional

import os
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from models.gaze_det.ptgaze.common.face_model_68 import FaceModel68
from models.gaze_det.ptgaze.common import Face, FacePartsName
from utils.estimator import Estimator
from utils.visualizer import Visualizer
from models.gaze_det.ptgaze.configs.default_config import get_config

import argparse
parser = argparse.ArgumentParser("Video Inference")
parser.add_argument("--input_path", "-i", type=str, required=True, help="Video input path")
parser.add_argument("--output_path", "-o", type=str, default='.', help="Video output path")
parser.add_argument("--detector", "-d", type=str, default="retinaface", help="detector")
parser.add_argument("--alignment", "-a", type=str, default="fan", help="alignment")
parser.add_argument("--gaze", "-z", type=str, default="ETH-XGaze", help="gaze")

class Demo:
    QUIT_KEYS = {27, ord("q")}

    def __init__(self, args):

        
        self.input_path = args.input_path
        self.output_path = args.output_path
        width, height = self.get_width_height(self.input_path)
        self.config = get_config(args.gaze, width, height)
        self.gaze_estimator = Estimator.from_name(
            det_name=args.detector,
            align_name=args.alignment,
            gaze_name=args.gaze,
            width=width , height= height
        )

        face_model_3d = FaceModel68()
        self.visualizer = Visualizer(
            self.gaze_estimator.camera, face_model_3d.NOSE_INDEX
        )

        self.cap = self._create_capture()
        self.output_dir = self._create_output_dir()
        self.writer = self._create_video_writer()

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_template_model = self.config.demo.show_template_model

    def get_width_height(self, path):
        extension = os.path.splitext(path)[1]
        if extension in ['.jpg', '.jpeg', '.png']:
            image = cv2.imread(path)
            h, w = image.shape[:2]
        elif extension in ['.mp4', '.avi']:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"{path} is not opened.")
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
        return w, h

    def run(self) -> None:
        extension = os.path.splitext(self.input_path)[1]
        if extension in ['.mp4', '.avi']:
            self._run_on_video()
        elif extension in ['.jpg', '.jpeg', '.png']:
            self._run_on_image()
        else:
            raise ValueError

    def _run_on_image(self):
        image = cv2.imread(self.input_path)
        self._process_image(image)
        if self.output_path:
            name = os.path.basename(self.input_path)
            output_path = pathlib.Path(self.output_path) / name
            cv2.imwrite(output_path.as_posix(), self.visualizer.image)

    def _run_on_video(self) -> None:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            self._process_image(frame)
        self.cap.release()
        if self.writer:
            self.writer.release()

    def _process_image(self, image) -> None:
        undistorted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        undistorted = cv2.undistort(
            undistorted,
            self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients,
        )
        undistorted = [undistorted]

        self.visualizer.set_image(image.copy())
        input, face = self.gaze_estimator.detect_faces(undistorted)
        self.gaze_estimator.estimate_gaze(input, face)
        self._draw_face_bbox(face)
        self._draw_head_pose(face)
        self._draw_landmarks(face)
        self._draw_face_template_model(face)
        self._draw_gaze_vector(face)

        self.visualizer.image = self.visualizer.image[:, ::-1]
        self.writer.write(self.visualizer.image)

    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.input_path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        return cap

    def _create_output_dir(self) -> Optional[pathlib.Path]:
        if not self.output_path:
            return
        output_dir = pathlib.Path(self.output_path)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @staticmethod
    def _create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime("%Y%m%d_%H%M%S")

    def _create_video_writer(self) -> Optional[cv2.VideoWriter]:

        extension = os.path.splitext(self.input_path)[1]
        if extension in ['.jpg', '.jpeg', '.png']:
            return None
        elif extension in ['.mp4', '.avi']:
            fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        else:
            raise ValueError
        
        name = pathlib.Path(self.input_path).stem
        FPS = int(self.cap.get(cv2.CAP_PROP_FPS))
        output_name = f"{name}.avi"
        output_path = self.output_dir / output_name
        writer = cv2.VideoWriter(
            output_path.as_posix(),
            fourcc,
            FPS,
            (self.gaze_estimator.camera.width, self.gaze_estimator.camera.height),
        )
        if writer is None:
            raise RuntimeError
        return writer

    def _wait_key(self) -> bool:
        key = cv2.waitKey(self.config.demo.wait_time) & 0xFF
        if key in self.QUIT_KEYS:
            self.stop = True
        elif key == ord("b"):
            self.show_bbox = not self.show_bbox
        elif key == ord("l"):
            self.show_landmarks = not self.show_landmarks
        elif key == ord("h"):
            self.show_head_pose = not self.show_head_pose
        elif key == ord("n"):
            self.show_normalized_image = not self.show_normalized_image
        elif key == ord("t"):
            self.show_template_model = not self.show_template_model
        else:
            return False
        return True

    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face: Face) -> None:
        if not self.show_head_pose:
            return
        # Draw the axes of the model coordinate system
        length = self.config.demo.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)

        euler_angles = face.head_pose_rot.as_euler("XYZ", degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        logger.info(
            f"[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, "
            f"roll: {roll:.2f}, distance: {face.distance:.2f}"
        )

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks, color=(0, 255, 255), size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d, color=(255, 0, 525), size=1)

    def _draw_gaze_vector(self, face: Face) -> None:
        length = self.config.demo.gaze_visualization_length
        if self.config.mode == "MPIIGaze":
            for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                eye = getattr(face, key.name.lower())
                self.visualizer.draw_3d_line(
                    eye.center, eye.center + length * eye.gaze_vector
                )
                pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
                logger.info(f"[{key.name.lower()}] pitch: {pitch:.2f}, yaw: {yaw:.2f}")
        elif self.config.mode in ["MPIIFaceGaze", "ETH-XGaze"]:
            self.visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector
            )
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            logger.info(f"[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}")
        else:
            raise ValueError


if __name__ == "__main__":

    args = parser.parse_args()
    demo = Demo(args)
    demo.run()
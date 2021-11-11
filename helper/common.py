import numpy as np
import cv2
import matplotlib.pyplot as plt


def plot_box(image, pred, figsize=(10, 10)):
    pred = np.squeeze(pred).tolist()
    box = [int(i) for i in pred]
    x, y, x2, y2 = box
    cv2.rectangle(image, (x, y), (x2, y2), color=(255, 0, 0), thickness=2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    plt.axis("off")
    plt.close()
    return fig


def _convert_pt(point: np.ndarray):
    return tuple(np.round(point).astype(np.int).tolist())


def draw_points(image, points, color=(255, 0, 0), size=3, figsize=(10, 10)) -> None:
    assert image is not None
    assert points.shape[1] == 2
    for pt in points:
        pt = _convert_pt(pt)
        cv2.circle(image, pt, size, color, cv2.FILLED)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)

    plt.axis("off")
    plt.close()
    return fig


def video_metadata(CAP):
    w = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    h = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    fps = int(CAP.get(cv2.CAP_PROP_FPS))
    total_frames = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT))
    return w, h, fps, total_frames

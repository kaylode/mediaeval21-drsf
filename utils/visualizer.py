import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_box(image, pred, figsize=(10,10)):
    pred = np.squeeze(pred).tolist()
    box = [int(i) for i in pred]
    x,y,x2,y2 = box
    cv2.rectangle(
        image, 
        (x,y), (x2,y2 ), 
        color=(255, 0, 0), thickness=2)

    fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    plt.axis('off')
    plt.close()
    return fig

def _convert_pt(point: np.ndarray):
    return tuple(np.round(point).astype(np.int).tolist())

def draw_points(
          image,
          points,
          color = (255, 0, 0),
          size = 3, figsize=(10,10)) -> None:
    assert image is not None
    assert points.shape[1] == 2
    for pt in points:
        pt = _convert_pt(pt)
        cv2.circle(image, pt, size, color, cv2.FILLED)

    fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    
    plt.axis('off')
    plt.close()
    return fig

def draw_3d_line(
        image,
        point0: np.ndarray,
        point1: np.ndarray,
        camera, 
        color = (0, 255, 0),
        lw=3,
    ) -> None:
        points3d = np.vstack([point0, point1])
        points2d = camera.project_points(points3d)
        pt0 = _convert_pt(points2d[0])
        pt1 = _convert_pt(points2d[1])
        image = cv2.line(image, pt0, pt1, color, lw, cv2.LINE_AA)
        return image


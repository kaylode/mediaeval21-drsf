from .common import draw_points, plot_box


class visualizer:
    def __init__(self, video):
        pass

    def draw(self, data):
        assert (
            "bboxes" in data.keys() or "points" in data.keys() or "gaze" in data.keys()
        ), "No data to draw"
        draw_keys = set("bboxes", "points", "gaze")

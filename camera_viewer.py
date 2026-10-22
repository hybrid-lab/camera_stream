"""Live viewer for the network camera (RTSP, H264DVR/Xiongmai-style firmware).

- Shows the stream as a single intensity channel through a selectable colormap (pyqtgraph/Qt).
- A vertical line cut at a chosen column, with its intensity profile plotted alongside.
- Digital averaging over the last N frames.

Usage:  python camera_viewer.py [--url RTSP_URL] [--ip 192.168.0.12]
"""
import argparse
import threading
import time
from collections import deque

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

DEFAULT_IP = "192.168.0.12"
# Default login for this firmware is admin with an empty password.
URL_TEMPLATE = "rtsp://{ip}:554/user=admin&password=&channel=1&stream={stream}.sdp?real_stream"
COLORMAPS = ["gray", "viridis", "inferno", "magma", "plasma", "turbo", "jet"]
MAX_AVG = 64


class FrameGrabber:
    """Reads the stream on a background thread, always keeping only the newest frame."""

    def __init__(self, url):
        self.url = url
        self._lock = threading.Lock()
        self._frame = None
        self._count = 0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print("Could not open stream, retrying...")
                time.sleep(2)
                continue
            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    print("Stream lost, reconnecting...")
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                with self._lock:
                    self._frame = gray
                    self._count += 1
            cap.release()

    def latest(self):
        """Return (frame, frame_counter); frame is None until the first one arrives."""
        with self._lock:
            return self._frame, self._count

    def stop(self):
        self._stop = True


class FrameAverager:
    """Running mean over the last n frames."""

    def __init__(self):
        self.n = 1
        self.reset()

    def reset(self):
        self._frames = deque()
        self._sum = None

    def set_n(self, n):
        self.n = n
        self.reset()

    def add(self, frame):
        f = frame.astype(np.float32)
        if self._sum is not None and self._sum.shape != f.shape:
            self.reset()
        if self._sum is None:
            self._sum = np.zeros_like(f)
        self._frames.append(f)
        self._sum += f
        while len(self._frames) > self.n:
            self._sum -= self._frames.popleft()
        return self._sum / len(self._frames)




def get_colormap(name):
    if name == "gray":
        return pg.ColorMap([0, 1], [(0, 0, 0), (255, 255, 255)])
    try:
        return pg.colormap.get(name)
    except Exception:
        return pg.colormap.get(name, source="matplotlib")


class Viewer(QtWidgets.QWidget):
    def __init__(self, url):
        super().__init__()
        self.setWindowTitle("Camera viewer")
        self.resize(1300, 780)
        self.grabber = FrameGrabber(url)
        self.avg = FrameAverager()
        self.last_count = 0
        self.shape = None
        self.data = None

        layout = QtWidgets.QVBoxLayout(self)
        glw = pg.GraphicsLayoutWidget()
        layout.addWidget(glw, 1)

        self.view = glw.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        self.img = pg.ImageItem(axisOrder="row-major", autoDownsample=True)
        self.view.addItem(self.img)
        self.vline = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("w", style=QtCore.Qt.PenStyle.DashLine))
        self.view.addItem(self.vline)
        self.vline.sigPositionChanged.connect(self._on_line)

        self.hist = pg.HistogramLUTItem(image=self.img)
        self.hist.setLevels(0, 255)
        glw.addItem(self.hist, row=0, col=1)

        self.prof_plot = glw.addPlot(row=0, col=2)
        self.prof_plot.invertY(True)
        self.prof_plot.setYLink(self.view)
        self.prof_plot.setLabel("bottom", "Intensity")
        self.prof_plot.setLabel("left", "Row (pixel)")
        self.prof_plot.setXRange(0, 255)
        self.prof_plot.showGrid(x=True, y=True, alpha=0.3)
        self.prof_curve = self.prof_plot.plot(pen="y")
        glw.ci.layout.setColumnStretchFactor(0, 4)
        glw.ci.layout.setColumnStretchFactor(2, 2)

        ctl = QtWidgets.QHBoxLayout()
        layout.addLayout(ctl)
        ctl.addWidget(QtWidgets.QLabel("Colormap"))
        self.cmap_box = QtWidgets.QComboBox()
        self.cmap_box.addItems(COLORMAPS)
        self.cmap_box.currentTextChanged.connect(self._set_cmap)
        ctl.addWidget(self.cmap_box)

        ctl.addWidget(QtWidgets.QLabel("Column x"))
        self.col_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.col_slider.setRange(0, 1)
        self.col_slider.valueChanged.connect(lambda v: self.vline.setValue(v))
        ctl.addWidget(self.col_slider, 2)
        self.col_label = QtWidgets.QLabel("0")
        self.col_label.setMinimumWidth(40)
        ctl.addWidget(self.col_label)

        ctl.addWidget(QtWidgets.QLabel("Avg frames"))
        self.avg_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.avg_slider.setRange(1, MAX_AVG)
        self.avg_slider.valueChanged.connect(self._on_avg)
        ctl.addWidget(self.avg_slider, 1)
        self.avg_label = QtWidgets.QLabel("1")
        self.avg_label.setMinimumWidth(30)
        ctl.addWidget(self.avg_label)

        self.cmap_box.setCurrentText("viridis")
        self._set_cmap("viridis")

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update)
        self.timer.start(10)

    def _set_cmap(self, name):
        self.hist.gradient.setColorMap(get_colormap(name))

    def _on_avg(self, n):
        self.avg.set_n(n)
        self.avg_label.setText(str(n))

    def _on_line(self):
        if self.shape is None:
            return
        col = int(round(min(max(self.vline.value(), 0), self.shape[1] - 1)))
        self.col_label.setText(str(col))
        if self.col_slider.value() != col:
            self.col_slider.blockSignals(True)
            self.col_slider.setValue(col)
            self.col_slider.blockSignals(False)
        self._draw_profile(col)

    def _draw_profile(self, col):
        if self.data is not None:
            prof = self.data[:, col]
            self.prof_curve.setData(prof, np.arange(len(prof)))

    def _update(self):
        frame, count = self.grabber.latest()
        if frame is None or count == self.last_count:
            return
        self.last_count = count
        self.data = self.avg.add(frame)
        if self.shape != frame.shape:
            self.shape = frame.shape
            h, w = self.shape
            self.col_slider.setRange(0, w - 1)
            self.vline.setBounds((0, w - 1))
            self.vline.setValue(w // 2)
            self.view.setRange(QtCore.QRectF(0, 0, w, h), padding=0)
            self.setWindowTitle(f"Camera viewer - {w} x {h}")
        self.img.setImage(self.data, autoLevels=False)
        self._on_line()

    def closeEvent(self, ev):
        self.grabber.stop()
        super().closeEvent(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=DEFAULT_IP)
    ap.add_argument("--stream", type=int, default=0, help="0 = main stream, 1 = sub stream")
    ap.add_argument("--url", help="full RTSP URL (overrides --ip/--stream)")
    args = ap.parse_args()
    url = args.url or URL_TEMPLATE.format(ip=args.ip, stream=args.stream)
    app = pg.mkQApp("Camera viewer")
    viewer = Viewer(url)
    viewer.show()
    app.exec()


if __name__ == "__main__":
    main()

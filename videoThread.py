#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Imports.
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from videoStream import VideoStream
import numpy as np

class VideoThread(QThread):
    # Signals enabling to update application from thread.
    changePixmapSignal = pyqtSignal(np.ndarray, QLabel)

    def __init__(self, args, imgLbl, vision3D):
        # Initialise.
        super().__init__()
        self._args = args
        self._imgLbl = imgLbl
        vision3D.changeParamSignal.connect(self.onParameterChanged)
        self._vid = VideoStream(self._args)

    @pyqtSlot(str, str)
    def onParameterChanged(self, param, value):
        self.stop() # Stop thread with previous capture (previous parameters).
        self._args[param] = int(value)
        self._vid = VideoStream(self._args)
        self.run() # Rerun thread with new capture (new parameters).

    def run(self):
        # Run thread.
        while self._vid.isOpened():
            frameOK, frame, _ = self._vid.read()
            if frameOK:
                self.changePixmapSignal.emit(frame, self._imgLbl)
        self._vid.release()

    def stop(self):
        # Stop thread.
        self._vid.release()
        self.wait()

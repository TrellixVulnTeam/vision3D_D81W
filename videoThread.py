#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Imports.
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import QThread, pyqtSignal
from videoStream import VideoStream
import numpy as np

class VideoThread(QThread):
    # Signals needed to update thread from application.
    changePixmapSignal = pyqtSignal(np.ndarray, QLabel)

    def __init__(self, args, imgLbl):
        # Initialise.
        super().__init__()
        self._runFlag = True
        self._args = args
        self._imgLbl = imgLbl

    def run(self):
        # Run thread.
        vid = VideoStream(self._args)
        while vid.isOpened() and self._runFlag:
            frameOK, frame, _ = vid.read()
            if frameOK:
                self.changePixmapSignal.emit(frame, self._imgLbl)
        vid.release()

    def stop(self):
        # Stop thread.
        self._runFlag = False
        self.wait()

#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Imports.
import sys
import argparse
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGridLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSlot, Qt
from videoThread import VideoThread
import cv2
import numpy as np

class Vision3D(QWidget):
    def __init__(self, args):
        # Initialise.
        super().__init__()
        self.setWindowTitle('Vision3D')

        # Create widgets.
        self.displayWidth = 640
        self.displayHeight = 480
        self.imgLblLeft = QLabel(self)
        self.imgLblLeft.resize(self.displayWidth, self.displayHeight)
        self.textLblLeft = QLabel('Left')
        self.imgLblRight = QLabel(self)
        self.imgLblRight.resize(self.displayWidth, self.displayHeight)
        self.textLblRight = QLabel('Right')

        # Handle alignment.
        self.textLblLeft.setAlignment(Qt.AlignCenter)
        self.imgLblLeft.setAlignment(Qt.AlignCenter)
        self.textLblRight.setAlignment(Qt.AlignCenter)
        self.imgLblRight.setAlignment(Qt.AlignCenter)

        # Handle layout.
        gridLay = QGridLayout()
        gridLay.addWidget(self.textLblLeft, 0, 0)
        gridLay.addWidget(self.imgLblLeft, 1, 0)
        gridLay.addWidget(self.textLblRight, 0, 1)
        gridLay.addWidget(self.imgLblRight, 1, 1)
        self.setLayout(gridLay)

        # Start threads.
        argsLeft = {key: val for key, val in args.items() if key != 'videoIDRight'}
        self.threadLeft = VideoThread(argsLeft, self.imgLblLeft)
        self.threadLeft.changePixmapSignal.connect(self.updateImage)
        self.threadLeft.start()
        argsRight = {key: val for key, val in args.items() if key != 'videoIDLeft'}
        self.threadRight = VideoThread(argsRight, self.imgLblRight)
        self.threadRight.changePixmapSignal.connect(self.updateImage)
        self.threadRight.start()

    def closeEvent(self, event):
        # Close application.
        self.threadLeft.stop()
        self.threadRight.stop()
        event.accept()

    @pyqtSlot(np.ndarray, QLabel)
    def updateImage(self, frame, imgLbl):
        # Update thread image.
        qtImg = self.convertCvQt(frame)
        imgLbl.setPixmap(qtImg)

    def convertCvQt(self, frame):
        # Convert frame to pixmap.
        rgbImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgbImg.shape
        bytesPerLine = channel * width
        qtImg = QImage(rgbImg.data, width, height, bytesPerLine, QImage.Format_RGB888)
        qtImg = qtImg.scaled(self.displayWidth, self.displayHeight, Qt.KeepAspectRatio)
        return QPixmap.fromImage(qtImg)

def cmdLineArgs():
    # Create parser.
    dscr = 'script designed for 3D vision.'
    parser = argparse.ArgumentParser(description=dscr)
    parser.add_argument('--hardware', type=str, required=True, metavar='HW',
                        choices=['arm-jetson', 'arm-nanopc', 'x86'],
                        help='select hardware to run on')
    parser.add_argument('--videoIDLeft', type=int, required=True, metavar='ID',
                        help='select left video stream to capture')
    parser.add_argument('--videoIDRight', type=int, required=True, metavar='ID',
                        help='select right video stream to capture')
    parser.add_argument('--videoType', type=str, default='CSI', metavar='T',
                        choices=['CSI', 'USB'],
                        help='select video type')
    parser.add_argument('--videoCapWidth', type=int, default=640, metavar='W',
                        help='define capture width')
    parser.add_argument('--videoCapHeight', type=int, default=360, metavar='H',
                        help='define capture height')
    parser.add_argument('--videoFrameRate', type=int, default=30, metavar='FR',
                        help='define frame rate')
    parser.add_argument('--videoFlipMethod', type=int, default=0, metavar='FM',
                        help='define flip method')
    parser.add_argument('--videoDspWidth', type=int, default=640, metavar='W',
                        help='define display width')
    parser.add_argument('--videoDspHeight', type=int, default=360, metavar='H',
                        help='define display height')
    args = parser.parse_args()

    return args

# Main program.
if __name__=="__main__":
    # Get command line arguments.
    args = cmdLineArgs()

   # Create Qt application.
    app = QApplication(sys.argv)
    v3D = Vision3D(vars(args))
    v3D.show()
    sys.exit(app.exec_())

#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Imports.
import sys
import argparse
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGridLayout
from PyQt5.QtWidgets import QGroupBox, QLineEdit
from PyQt5.QtGui import QImage, QPixmap, QIntValidator
from PyQt5.QtCore import pyqtSlot, Qt, pyqtSignal
from videoThread import VideoThread
import cv2
import numpy as np

class Vision3DEdit(QLineEdit):
    def __init__(self, param, changeParamSignal):
        # Initialise.
        super().__init__()
        self.edt = QLineEdit()
        self._param = param # Track parameter associated to QLineEdit.
        self._changeParamSignal = changeParamSignal

    def onParameterChanged(self):
        # Callback on parameter change.
        value = self.edt.text() # Text which has been modified.
        self._changeParamSignal.emit(self._param, value) # Emit value and associated parameter.

class Vision3D(QWidget):
    # Signals enabling to update thread from application.
    changeParamSignal = pyqtSignal(str, str)

    def __init__(self, args):
        # Initialise.
        super().__init__()
        self.setWindowTitle('Vision3D')

        # Create widgets.
        self.displayWidth = 640
        self.displayHeight = 480
        grpBox = QGroupBox('Parameters')
        grpBoxLay = QGridLayout()
        grpBox.setLayout(grpBoxLay)
        self._edtParams = [] # Vision3DEdit instances lifecycle MUST be consistent with Vision3D lifecycle.
        self._createParameters(grpBoxLay, 'videoCapWidth', args['videoCapWidth'], 0, 1)
        self._createParameters(grpBoxLay, 'videoCapHeight', args['videoCapHeight'], 0, 2)
        self._createParameters(grpBoxLay, 'videoCapFrameRate', args['videoCapFrameRate'], 0, 3)
        self._createParameters(grpBoxLay, 'videoFlipMethod', args['videoFlipMethod'], 0, 4)
        self._createParameters(grpBoxLay, 'videoDspWidth', args['videoDspWidth'], 0, 5)
        self._createParameters(grpBoxLay, 'videoDspHeight', args['videoDspHeight'], 0, 6)
        self.imgLblLeft = QLabel(self)
        self.imgLblLeft.resize(self.displayWidth, self.displayHeight)
        self.txtLblLeft = QLabel('Left')
        self.imgLblRight = QLabel(self)
        self.imgLblRight.resize(self.displayWidth, self.displayHeight)
        self.txtLblRight = QLabel('Right')

        # Handle alignment.
        self.txtLblLeft.setAlignment(Qt.AlignCenter)
        self.imgLblLeft.setAlignment(Qt.AlignCenter)
        self.txtLblRight.setAlignment(Qt.AlignCenter)
        self.imgLblRight.setAlignment(Qt.AlignCenter)

        # Handle layout.
        grdLay = QGridLayout()
        grdLay.addWidget(grpBox, 0, 0, 1, 2)
        grdLay.addWidget(self.txtLblLeft, 1, 0)
        grdLay.addWidget(self.txtLblRight, 1, 1)
        grdLay.addWidget(self.imgLblLeft, 2, 0)
        grdLay.addWidget(self.imgLblRight, 2, 1)
        self.setLayout(grdLay)

        # Start threads.
        argsLeft = {key: val for key, val in args.items() if key != 'videoIDRight'}
        self.threadLeft = VideoThread(argsLeft, self.imgLblLeft, self.txtLblLeft, self)
        self.threadLeft.changePixmapSignal.connect(self.updateImage)
        self.threadLeft.start()
        argsRight = {key: val for key, val in args.items() if key != 'videoIDLeft'}
        self.threadRight = VideoThread(argsRight, self.imgLblRight, self.txtLblRight, self)
        self.threadRight.changePixmapSignal.connect(self.updateImage)
        self.threadRight.start()

    def _createParameters(self, grpBoxLay, param, val, row, col, enable=False):
        # Create one parameter.
        lbl = QLabel(param)
        v3DEdt = Vision3DEdit(param, self.changeParamSignal)
        v3DEdt.edt.setValidator(QIntValidator())
        v3DEdt.edt.setEnabled(enable)
        v3DEdt.edt.setText(str(val))
        v3DEdt.edt.editingFinished.connect(v3DEdt.onParameterChanged)
        grdLay = QGridLayout()
        grdLay.addWidget(lbl, 0, 0)
        grdLay.addWidget(v3DEdt.edt, 0, 1)
        grpBoxLay.addLayout(grdLay, row, col)
        self._edtParams.append(v3DEdt) # Vision3DEdit instances lifecycle MUST be consistent with Vision3D lifecycle.

    def closeEvent(self, event):
        # Close application.
        self.threadLeft.stop()
        self.threadRight.stop()
        for v3DEdt in self._edtParams:
            v3DEdt.close() # Vision3DEdit instances lifecycle MUST be consistent with Vision3D lifecycle.
        event.accept()

    @pyqtSlot(np.ndarray, QLabel, int, QLabel)
    def updateImage(self, frame, imgLbl, fps, txtLbl):
        # Update thread image.
        qtImg = self.convertCvQt(frame)
        imgLbl.setPixmap(qtImg)
        txt = txtLbl.text()
        lbl = txt.split()[0] # Suppress old FPS: retrive only first word (left/right).
        txtLbl.setText(lbl + ' - FPS %d'%fps)

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
    parser.add_argument('--videoCapFrameRate', type=int, default=30, metavar='FR',
                        help='define capture frame rate')
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

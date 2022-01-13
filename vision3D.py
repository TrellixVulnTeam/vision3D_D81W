#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Imports.
import sys
import argparse
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGridLayout
from PyQt5.QtWidgets import QGroupBox, QLineEdit, QCheckBox, QRadioButton
from PyQt5.QtGui import QImage, QPixmap, QIntValidator, QDoubleValidator
from PyQt5.QtCore import pyqtSlot, Qt, pyqtSignal
from videoThread import VideoThread
import cv2
import numpy as np
import threading

class Vision3DEdit(QLineEdit):
    def __init__(self, param, objType, vision3D):
        # Initialise.
        super().__init__()
        self.edt = QLineEdit()
        self._param = param # Track parameter associated to QLineEdit.
        self._objType = objType
        self._vision3D = vision3D

    def onParameterChanged(self):
        # Callback on parameter change.
        value = self.edt.text() # Text which has been modified.
        self._vision3D.changeParamSignal.emit(self._param, self._objType, value) # Emit value and associated parameter / type.
        if self.edt.isEnabled():
            self._vision3D.disableCalibration()

class Vision3DCheckBox(QCheckBox):
    def __init__(self, param, vision3D):
        # Initialise.
        super().__init__()
        self.chkBox = QCheckBox()
        self._param = param # Track parameter associated to QLineEdit.
        self._vision3D = vision3D

    def onParameterChanged(self):
        # Callback on parameter change.
        value = self.chkBox.isChecked() # State which has been modified.
        self._vision3D.changeParamSignal.emit(self._param, 'bool', value) # Emit value and associated parameter / type.

class Vision3DRadioButton(QRadioButton):
    def __init__(self, param, vision3D):
        # Initialise.
        super().__init__()
        self.rdoBoxRaw = QRadioButton('raw')
        self.rdoBoxRaw.mode = 'raw'
        self.rdoBoxUnd = QRadioButton('undistort')
        self.rdoBoxUnd.mode = 'und'
        self.rdoBoxStr = QRadioButton('stereo')
        self.rdoBoxStr.mode = 'str'
        self._param = param # Track parameter associated to QLineEdit.
        self._vision3D = vision3D

    def onParameterChanged(self):
        # Callback on parameter change.
        rdoBtn = self.sender()
        if rdoBtn.isChecked():
            value = rdoBtn.mode # Mode which has been modified.
            self._vision3D.changeParamSignal.emit(self._param, 'str', value) # Emit value and associated parameter / type.
            self._vision3D.disableCalibration()

class Vision3D(QWidget):
    # Signals enabling to update thread from application.
    changeParamSignal = pyqtSignal(str, str, object) # object may be int, double, ...
    lock = threading.Lock()
    calibratedThreads = 0

    def __init__(self, args):
        # Initialise.
        super().__init__()
        self.setWindowTitle('Vision3D')

        # Create parameters.
        self._args = args.copy()
        self._args['alpha'] = 0.
        self._args['ROI'] = False
        self._args['mode'] = 'raw'
        grpBox = QGroupBox('Parameters')
        grpBoxLay = QGridLayout()
        grpBox.setLayout(grpBoxLay)
        self._edtParams = []
        self._createEditParameters(grpBoxLay, 'videoCapWidth', 0, 1)
        self._createEditParameters(grpBoxLay, 'videoCapHeight', 0, 2)
        self._createEditParameters(grpBoxLay, 'videoCapFrameRate', 0, 3)
        if self._args['hardware'] == 'arm-jetson':
            self._createEditParameters(grpBoxLay, 'videoFlipMethod', 1, 1)
            self._createEditParameters(grpBoxLay, 'videoDspWidth', 1, 2)
            self._createEditParameters(grpBoxLay, 'videoDspHeight', 1, 3)
        self._createRdoButParameters(grpBoxLay, 'mode', 0, 4)
        self._createEditParameters(grpBoxLay, 'alpha', 1, 4, enable=True, objType='double')
        self._createChkBoxParameters(grpBoxLay, 'ROI', 1, 5)

        # Create widgets.
        self.imgLblLeft = QLabel(self)
        self.txtLblLeft = QLabel('Left')
        self.imgLblRight = QLabel(self)
        self.txtLblRight = QLabel('Right')
        self._resizeFrames()

        # Handle alignment.
        grpBox.setAlignment(Qt.AlignCenter)
        grpBoxLay.setAlignment(Qt.AlignCenter)
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
        videoIDLeft = args['videoIDLeft']
        self._threadLeft = VideoThread(videoIDLeft, self._args, self.imgLblLeft, self.txtLblLeft, self)
        self._threadLeft.changePixmapSignal.connect(self.updateFrame)
        self._threadLeft.calibrationDoneSignal.connect(self.calibrationDone)
        self._threadLeft.start()
        videoIDRight = args['videoIDRight']
        self._threadRight = VideoThread(videoIDRight, self._args, self.imgLblRight, self.txtLblRight, self)
        self._threadRight.changePixmapSignal.connect(self.updateFrame)
        self._threadRight.calibrationDoneSignal.connect(self.calibrationDone)
        self._threadRight.start()

    def _createEditParameters(self, grpBoxLay, param, row, col, enable=False, objType='int'):
        # Create one parameter.
        lbl = QLabel(param)
        v3DEdt = Vision3DEdit(param, objType, self)
        if objType == 'int':
            v3DEdt.edt.setValidator(QIntValidator())
        elif objType == 'double':
            v3DEdt.edt.setValidator(QDoubleValidator())
        val = self._args[param]
        v3DEdt.edt.setText(str(val))
        v3DEdt.edt.editingFinished.connect(v3DEdt.onParameterChanged)
        grdLay = QGridLayout()
        grdLay.addWidget(lbl, 0, 0)
        grdLay.addWidget(v3DEdt.edt, 0, 1)
        grpBoxLay.addLayout(grdLay, row, col)
        v3DEdt.edt.setEnabled(enable)
        if enable: # This is a GUI that can be potentially controled.
            self._edtParams.append(v3DEdt) # GUI controls lifecycle MUST be consistent with Vision3D lifecycle.

    def _createChkBoxParameters(self, grpBoxLay, param, row, col):
        # Create one parameter.
        lbl = QLabel(param)
        self.v3DChkBox = Vision3DCheckBox(param, self)
        val = self._args[param]
        self.v3DChkBox.chkBox.setCheckState(val)
        self.v3DChkBox.chkBox.toggled.connect(self.v3DChkBox.onParameterChanged)
        grdLay = QGridLayout()
        grdLay.addWidget(lbl, 0, 0)
        grdLay.addWidget(self.v3DChkBox.chkBox, 0, 1)
        grpBoxLay.addLayout(grdLay, row, col)

    def _createRdoButParameters(self, grpBoxLay, param, row, col):
        # Create one parameter.
        lbl = QLabel(param)
        self.v3DRdoBtn = Vision3DRadioButton(param, self)
        self.v3DRdoBtn.rdoBoxRaw.setChecked(True)
        self.v3DRdoBtn.rdoBoxUnd.setChecked(False)
        self.v3DRdoBtn.rdoBoxStr.setChecked(False)
        self.v3DRdoBtn.rdoBoxRaw.toggled.connect(self.v3DRdoBtn.onParameterChanged)
        self.v3DRdoBtn.rdoBoxUnd.toggled.connect(self.v3DRdoBtn.onParameterChanged)
        self.v3DRdoBtn.rdoBoxStr.toggled.connect(self.v3DRdoBtn.onParameterChanged)
        grdLay = QGridLayout()
        grdLay.addWidget(lbl, 0, 0)
        grdLay.addWidget(self.v3DRdoBtn.rdoBoxRaw, 0, 1)
        grdLay.addWidget(self.v3DRdoBtn.rdoBoxUnd, 0, 2)
        grdLay.addWidget(self.v3DRdoBtn.rdoBoxStr, 0, 3)
        grpBoxLay.addLayout(grdLay, row, col, 1, 2)

    def _getFrameSize(self):
        # Get frame size.
        displayHeight, displayWidth = 0, 0
        if self._args['hardware'] == 'arm-jetson':
            displayHeight = self._args['videoDspHeight']
            displayWidth = self._args['videoDspWidth']
        else:
            displayHeight = self._args['videoCapHeight']
            displayWidth = self._args['videoCapWidth']
        return displayHeight, displayWidth

    def _resizeFrames(self):
        # Resize images.
        displayHeight, displayWidth = self._getFrameSize()
        self.imgLblLeft.resize(displayWidth, displayHeight)
        self.imgLblRight.resize(displayWidth, displayHeight)

    def closeEvent(self, event):
        # Close application.
        self._threadLeft.stop()
        self._threadRight.stop()
        for v3DEdt in self._edtParams:
            v3DEdt.close() # GUI controls lifecycle MUST be consistent with Vision3D lifecycle.
        self.v3DChkBox.close() # GUI controls lifecycle MUST be consistent with Vision3D lifecycle.
        self.v3DRdoBtn.close() # GUI controls lifecycle MUST be consistent with Vision3D lifecycle.
        event.accept()

    @pyqtSlot(np.ndarray, QLabel, int, QLabel)
    def updateFrame(self, frame, imgLbl, fps, txtLbl):
        # Update thread image.
        qtImg = self.convertCvQt(frame)
        imgLbl.setPixmap(qtImg)

        # Update thread label.
        txt = txtLbl.text()
        lbl = txt.split()[0] # Suppress old FPS: retrive only first word (left/right).
        txtLbl.setText(lbl + ' - FPS %d'%fps)

    def convertCvQt(self, frame):
        # Convert frame to pixmap.
        rgbImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        displayHeight, displayWidth, channel = rgbImg.shape
        bytesPerLine = channel * displayWidth
        qtImg = QImage(rgbImg.data, displayWidth, displayHeight, bytesPerLine, QImage.Format_RGB888)
        return QPixmap.fromImage(qtImg)

    @pyqtSlot(int)
    def calibrationDone(self, vidID):
        # Update calibrated thread status.
        self.lock.acquire()
        self.calibratedThreads += vidID
        self.lock.release()

        # Re-enable radio buttons when both threads are calibrated.
        if self.calibratedThreads == self._threadLeft.vidID + self._threadRight.vidID:
            self.calibratedThreads = 0
            self.v3DRdoBtn.rdoBoxRaw.setEnabled(True)
            self.v3DRdoBtn.rdoBoxUnd.setEnabled(True)
            self.v3DRdoBtn.rdoBoxStr.setEnabled(True)
            for v3DEdt in self._edtParams:
                v3DEdt.edt.setEnabled(True)

    def disableCalibration(self):
        # Disable access to calibration parameters to prevent thread overflow.
        self.v3DRdoBtn.rdoBoxRaw.setEnabled(False)
        self.v3DRdoBtn.rdoBoxUnd.setEnabled(False)
        self.v3DRdoBtn.rdoBoxStr.setEnabled(False)
        for v3DEdt in self._edtParams:
            v3DEdt.edt.setEnabled(False)

        # Black out frames.
        displayHeight, displayWidth = self._getFrameSize()
        shape = (displayHeight, displayWidth)
        frame = np.ones(shape, np.uint8) # Black image.
        self.updateFrame(frame, self.imgLblLeft, 0, self.txtLblLeft)
        self.updateFrame(frame, self.imgLblRight, 0, self.txtLblRight)

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
    parser.add_argument('--debug', dest='debug', action='store_true')
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

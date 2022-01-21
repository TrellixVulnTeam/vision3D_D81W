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

class Vision3DEdit(QWidget):
    def __init__(self, param, objType, parent=None):
        # Initialise.
        super().__init__(parent)
        self.gui = QLineEdit()
        self._param = param # Track associated parameter.
        self._objType = objType
        self._vision3D = parent

    def onParameterChanged(self):
        # Callback on parameter change.
        if self.gui.isEnabled():
            self._vision3D.disableCalibration()
        value = self.gui.text() # Text which has been modified.
        self._vision3D.changeParamSignal.emit(self._param, self._objType, value) # Emit value and associated parameter / type.

class Vision3DCheckBox(QWidget):
    def __init__(self, param, triggerDisable, parent=None):
        # Initialise.
        super().__init__(parent)
        self.gui = QCheckBox()
        self._param = param # Track associated parameter.
        self._triggerDisable = triggerDisable # Need to disable calibration on callback.
        self._vision3D = parent

    def onParameterChanged(self):
        # Callback on parameter change.
        if self._triggerDisable:
            self._vision3D.disableCalibration()
        value = self.gui.isChecked() # State which has been modified.
        self._vision3D.changeParamSignal.emit(self._param, 'bool', value) # Emit value and associated parameter / type.

class Vision3DRadioButton(QWidget):
    def __init__(self, param, parent=None):
        # Initialise.
        super().__init__(parent)
        self.rdoBoxRaw = QRadioButton('raw')
        self.rdoBoxRaw.mode = 'raw'
        self.rdoBoxUnd = QRadioButton('undistort')
        self.rdoBoxUnd.mode = 'und'
        self.rdoBoxStr = QRadioButton('stereo')
        self.rdoBoxStr.mode = 'str'
        self._param = param # Track associated parameter.
        self._vision3D = parent

    def onParameterChanged(self):
        # Callback on parameter change.
        rdoBtn = self.sender()
        if rdoBtn.isChecked():
            self._vision3D.disableCalibration()
            value = rdoBtn.mode # Mode which has been modified.
            self._vision3D.changeParamSignal.emit(self._param, 'str', value) # Emit value and associated parameter / type.

class Vision3D(QWidget):
    # Signals enabling to update thread from application.
    changeParamSignal = pyqtSignal(str, str, object) # object may be int, double, ...
    calibratedThreadsLock = threading.Lock()
    calibratedThreads = 0

    def __init__(self, args):
        # Initialise.
        super().__init__()
        self.setWindowTitle('Vision3D')

        # Create parameters.
        self._args = args.copy()
        self._args['alpha'] = 0.
        self._args['fovScale'] = 1.
        self._args['balance'] = 0.
        self._args['CAL'] = False
        self._args['ROI'] = False
        self._args['DBG'] = False
        self._args['mode'] = 'raw'
        grpBox = QGroupBox('Parameters')
        grpBoxLay = QGridLayout()
        grpBox.setLayout(grpBoxLay)
        self._guiCtrParams = [] # All GUIs with controls.
        self._createEditParameters(grpBoxLay, 'videoCapWidth', 1, 0)
        self._createEditParameters(grpBoxLay, 'videoCapHeight', 1, 1)
        self._createEditParameters(grpBoxLay, 'videoCapFrameRate', 1, 2)
        if self._args['hardware'] == 'arm-jetson':
            self._createEditParameters(grpBoxLay, 'videoFlipMethod', 2, 0)
            self._createEditParameters(grpBoxLay, 'videoDspWidth', 2, 1)
            self._createEditParameters(grpBoxLay, 'videoDspHeight', 2, 2)
        self._createRdoButParameters(grpBoxLay, 'mode', 0, 6)
        if self._args['fisheye']:
            tooltip = 'Divisor for new focal length.'
            self._createEditParameters(grpBoxLay, 'fovScale', 2, 7, rowSpan=2, colSpan=1,
                                       enable=True, objType='double', tooltip=tooltip)
            tooltip = 'Sets the new focal length in range between the min focal length and the max\n'
            tooltip += 'focal length. Balance is in range of [0, 1].'
            self._createEditParameters(grpBoxLay, 'balance', 2, 8, rowSpan=2, colSpan=1,
                                       enable=True, objType='double', tooltip=tooltip)
        else:
            self._createChkBoxParameters(grpBoxLay, 'CAL', 3, 7, triggerDisable=True)
            tooltip = 'Free scaling parameter between 0 and 1.\n'
            tooltip += '  - 0: rectified images are zoomed and shifted so\n'
            tooltip += '       that only valid pixels are visible: no black areas after rectification\n'
            tooltip += '  - 1: rectified image is decimated and shifted so that all the pixels from the\n'
            tooltip += '       original images from the cameras are retained in the rectified images:\n'
            tooltip += '       no source image pixels are lost.\n'
            tooltip += 'Any intermediate value yields an intermediate result between those two extreme cases.\n'
            tooltip += 'If negative, the function performs the default scaling.'
            self._createEditParameters(grpBoxLay, 'alpha', 2, 8, rowSpan=2, colSpan=1,
                                       enable=True, objType='double', tooltip=tooltip)
        self._ckbROI = self._createChkBoxParameters(grpBoxLay, 'ROI', 2, 9, rowSpan=2, colSpan=1)
        self._createChkBoxParameters(grpBoxLay, 'DBG', 0, 9)

        # Create widgets.
        self.imgLblLeft = QLabel()
        self.txtLblLeft = QLabel('Left')
        self.imgLblRight = QLabel()
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

    def _createEditParameters(self, grpBoxLay, param, row, col, rowSpan=1, colSpan=1,
                              enable=False, objType='int', tooltip=None):
        # Create one parameter.
        lbl = QLabel(param)
        v3DEdt = Vision3DEdit(param, objType, parent=self)
        if objType == 'int':
            v3DEdt.gui.setValidator(QIntValidator())
        elif objType == 'double':
            v3DEdt.gui.setValidator(QDoubleValidator())
        val = self._args[param]
        v3DEdt.gui.setText(str(val))
        v3DEdt.gui.editingFinished.connect(v3DEdt.onParameterChanged)
        grpBoxLay.addWidget(lbl, row, 2*col+0, rowSpan, colSpan)
        grpBoxLay.addWidget(v3DEdt.gui, row, 2*col+1, rowSpan, colSpan)
        v3DEdt.gui.setEnabled(enable)
        if enable:
            self._guiCtrParams.append(v3DEdt) # Enabled edits have controls.
        if tooltip:
            lbl.setToolTip(tooltip)

    def _createChkBoxParameters(self, grpBoxLay, param, row, col, triggerDisable=False, rowSpan=1, colSpan=1):
        # Create one parameter.
        lbl = QLabel(param)
        v3DChkBox = Vision3DCheckBox(param, triggerDisable, parent=self)
        val = self._args[param]
        v3DChkBox.gui.setCheckState(val)
        v3DChkBox.gui.toggled.connect(v3DChkBox.onParameterChanged)
        grpBoxLay.addWidget(lbl, row, 2*col+0, rowSpan, colSpan)
        grpBoxLay.addWidget(v3DChkBox.gui, row, 2*col+1, rowSpan, colSpan)
        self._guiCtrParams.append(v3DChkBox) # Enabled checkbox may have controls.
        return v3DChkBox

    def _createRdoButParameters(self, grpBoxLay, param, row, col):
        # Create one parameter.
        lbl = QLabel(param)
        self.v3DRdoBtn = Vision3DRadioButton(param, parent=self)
        self.v3DRdoBtn.rdoBoxRaw.setChecked(True)
        self.v3DRdoBtn.rdoBoxUnd.setChecked(False)
        self.v3DRdoBtn.rdoBoxStr.setChecked(False)
        self.v3DRdoBtn.rdoBoxRaw.toggled.connect(self.v3DRdoBtn.onParameterChanged)
        self.v3DRdoBtn.rdoBoxUnd.toggled.connect(self.v3DRdoBtn.onParameterChanged)
        self.v3DRdoBtn.rdoBoxStr.toggled.connect(self.v3DRdoBtn.onParameterChanged)
        grpBoxLay.addWidget(lbl, row+0, col)
        grpBoxLay.addWidget(self.v3DRdoBtn.rdoBoxRaw, row+1, col)
        grpBoxLay.addWidget(self.v3DRdoBtn.rdoBoxUnd, row+2, col)
        grpBoxLay.addWidget(self.v3DRdoBtn.rdoBoxStr, row+3, col)

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

    @pyqtSlot(int, bool)
    def calibrationDone(self, vidID, hasROI):
        # Update calibrated thread status.
        self.calibratedThreadsLock.acquire()
        self.calibratedThreads += vidID
        self.calibratedThreadsLock.release()

        # Re-enable radio buttons when both threads are calibrated.
        if self.calibratedThreads == self._threadLeft.vidID + self._threadRight.vidID:
            self.calibratedThreads = 0
            self.v3DRdoBtn.rdoBoxRaw.setEnabled(True)
            self.v3DRdoBtn.rdoBoxUnd.setEnabled(True)
            self.v3DRdoBtn.rdoBoxStr.setEnabled(True)
            for v3DEdt in self._guiCtrParams:
                v3DEdt.gui.setEnabled(True)
            self._ckbROI.gui.setEnabled(hasROI)

    def disableCalibration(self):
        # Disable access to calibration parameters to prevent thread overflow.
        self.v3DRdoBtn.rdoBoxRaw.setEnabled(False)
        self.v3DRdoBtn.rdoBoxUnd.setEnabled(False)
        self.v3DRdoBtn.rdoBoxStr.setEnabled(False)
        for v3DEdt in self._guiCtrParams:
            v3DEdt.gui.setEnabled(False)
        self._ckbROI.gui.setEnabled(False)

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
    parser.add_argument('--fisheye', dest='fisheye', action='store_true',
                        help='use fisheye cameras.')
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

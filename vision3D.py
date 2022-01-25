#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Imports.
import sys
import os
import wget
from google_drive_downloader import GoogleDriveDownloader as gdd
import tarfile
import argparse
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGridLayout
from PyQt5.QtWidgets import QGroupBox, QLineEdit, QCheckBox, QRadioButton, QButtonGroup
from PyQt5.QtGui import QImage, QPixmap, QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt, pyqtSignal, QThreadPool, QObject
from videoThread import VideoThread
import cv2
import numpy as np
import threading
from videoStream import cmdLineArgsVideoStream
from calibrate import cmdLineArgsCalibrate
import logging

logger = logging.getLogger()

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
        self._vision3D.disableCalibration()
        value = self.gui.text() # Text which has been modified.
        self._vision3D.signals.changeParam.emit(self._param, self._objType, value) # Emit value and associated parameter / type.

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
        self._vision3D.signals.changeParam.emit(self._param, 'bool', value) # Emit value and associated parameter / type.

class Vision3DRadioButtonMode(QWidget):
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
        grpBtn = QButtonGroup(parent)
        grpBtn.setExclusive(True) # Make radio button exclusive.
        grpBtn.addButton(self.rdoBoxRaw)
        grpBtn.addButton(self.rdoBoxUnd)
        grpBtn.addButton(self.rdoBoxStr)

    def onParameterChanged(self):
        # Callback on parameter change.
        rdoBtn = self.sender()
        if rdoBtn.isChecked():
            # Send signal to threads.
            self._vision3D.disableCalibration()
            value = rdoBtn.mode # Mode which has been modified.
            self._vision3D.signals.changeParam.emit(self._param, 'str', value) # Emit value and associated parameter / type.

class Vision3DRadioButtonDetection(QWidget):
    def __init__(self, param, parent=None):
        # Initialise.
        super().__init__(parent)
        self.rdoBoxNone = QRadioButton('None')
        self.rdoBoxNone.mode = 'None'
        self.rdoBoxYOLO = QRadioButton('YOLO')
        self.rdoBoxYOLO.mode = 'YOLO'
        self.rdoBoxSSD = QRadioButton('SSD')
        self.rdoBoxSSD.mode = 'SSD'
        self._param = param # Track associated parameter.
        self._vision3D = parent
        grpBtn = QButtonGroup(parent)
        grpBtn.setExclusive(True) # Make radio button exclusive.
        grpBtn.addButton(self.rdoBoxNone)
        grpBtn.addButton(self.rdoBoxYOLO)
        grpBtn.addButton(self.rdoBoxSSD)

    def onParameterChanged(self):
        # Callback on parameter change.
        rdoBtn = self.sender()
        if rdoBtn.isChecked():
            # Send signal to threads.
            self._vision3D.disableCalibration()
            value = rdoBtn.mode # Mode which has been modified.
            self._vision3D.signals.changeParam.emit(self._param, 'str', value) # Emit value and associated parameter / type.

class Vision3DSignals(QObject):
    # Signals enabling to update threads from application.
    changeParam = pyqtSignal(str, str, object) # May be int, double, ...
    stop = pyqtSignal() # GUI is closed.

class Vision3D(QWidget):
    # Signals enabling to update thread from application.
    calibratedThreadsLock = threading.Lock()
    calibratedThreads = 0

    def __init__(self, args):
        # Initialise.
        super().__init__()
        self.setWindowTitle('Vision3D')

        # Set up info/debug log on demand.
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%H:%M:%S', level=logging.INFO)

        # Create parameters.
        self._args = args.copy()
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
        self._args['mode'] = 'raw'
        self._createRdoButMode(grpBoxLay, 'mode', 0, 6)
        if self._args['fisheye']:
            tooltip = 'Divisor for new focal length.'
            self._createEditParameters(grpBoxLay, 'fovScale', 2, 7, rowSpan=2, colSpan=1,
                                       enable=True, objType='double', tooltip=tooltip)
            tooltip = 'Sets the new focal length in range between the min focal length and the max\n'
            tooltip += 'focal length. Balance is in range of [0, 1].'
            self._createEditParameters(grpBoxLay, 'balance', 2, 8, rowSpan=2, colSpan=1,
                                       enable=True, objType='double', tooltip=tooltip)
        else:
            self._args['CAL'] = False
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
        self._args['ROI'] = False
        self._ckbROI = self._createChkBoxParameters(grpBoxLay, 'ROI', 2, 9, rowSpan=2, colSpan=1)
        self._args['detection'] = 'None'
        self._createRdoButDetection(grpBoxLay, 'detection', 0, 20)
        self._args['confidence'] = 0.5
        self._createEditParameters(grpBoxLay, 'confidence', 0, 24, enable=True, objType='double')
        self._args['nms'] = 0.3
        self._createEditParameters(grpBoxLay, 'nms', 0, 25, enable=True, objType='double')
        self._args['DBG'] = False
        self._createChkBoxParameters(grpBoxLay, 'DBG', 4, 26)

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

        # Download COCO dataset labels.
        if not os.path.isfile('coco.names'):
            wget.download('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names')
        else:
            logger.info('[vision3D] coco.names has already been downloaded.')

        # Download YOLO files.
        if not os.path.isfile('yolov3-tiny.weights'):
            wget.download('https://pjreddie.com/media/files/yolov3-tiny.weights')
        else:
            logger.info('[vision3D] yolov3-tiny.weights has already been downloaded.')
        if not os.path.isfile('yolov3-tiny.cfg'):
            wget.download('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg')
        else:
            logger.info('[vision3D] yolov3-tiny.cfg has already been downloaded.')

        # Download Model COCO SSD512 file: https://github.com/weiliu89/caffe/tree/ssd (fork from BVLC/caffe).
        if not os.path.isfile('models_VGGNet_coco_SSD_512x512.tar.gz'):
            gdd.download_file_from_google_drive(file_id='0BzKzrI_SkD1_dlJpZHJzOXd3MTg',
                                                dest_path='./models_VGGNet_coco_SSD_512x512.tar.gz',
                                                showsize=True, unzip=False)
            tgz = tarfile.open('models_VGGNet_coco_SSD_512x512.tar.gz')
            tgz.extractall('./models_VGGNet_coco_SSD_512x512')
            tgz.close()
        else:
            logger.info('[vision3D] models_VGGNet_coco_SSD_512x512.tar.gz has already been downloaded.')

        # Start threads.
        self.signals = Vision3DSignals()
        self._threadPool = QThreadPool() # QThreadPool must be used with QRunnable (NOT QThread).
        self._threadPool.setMaxThreadCount(2)
        videoIDLeft = args['videoIDLeft']
        self._threadLeft = VideoThread(videoIDLeft, self._args, self.imgLblLeft, self.txtLblLeft, self)
        self._threadLeft.signals.changePixmap.connect(self.updateFrame)
        self._threadLeft.signals.calibrationDone.connect(self.calibrationDone)
        self._threadPool.start(self._threadLeft)
        videoIDRight = args['videoIDRight']
        self._threadRight = VideoThread(videoIDRight, self._args, self.imgLblRight, self.txtLblRight, self)
        self._threadRight.signals.changePixmap.connect(self.updateFrame)
        self._threadRight.signals.calibrationDone.connect(self.calibrationDone)
        self._threadPool.start(self._threadRight)

    def updateFrame(self, frame, imgLbl, fps, txtLbl):
        # Update thread image.
        qtImg = self._convertCvQt(frame)
        imgLbl.setPixmap(qtImg)

        # Update thread label.
        txt = txtLbl.text()
        lbl = txt.split()[0] # Suppress old FPS: retrive only first word (left/right).
        txtLbl.setText(lbl + ' - FPS %d'%fps)

    def calibrationDone(self, vidID, hasROI):
        # Re-enable radio buttons when both threads are calibrated.
        self.calibratedThreadsLock.acquire()
        self.calibratedThreads += vidID
        if self.calibratedThreads == self._threadLeft.vidID + self._threadRight.vidID:
            self.calibratedThreads = 0
            self.v3DRdoBtnMode.rdoBoxRaw.setEnabled(True)
            self.v3DRdoBtnMode.rdoBoxUnd.setEnabled(True)
            self.v3DRdoBtnMode.rdoBoxStr.setEnabled(True)
            self.v3DRdoBtnDetect.rdoBoxNone.setEnabled(True)
            self.v3DRdoBtnDetect.rdoBoxYOLO.setEnabled(True)
            self.v3DRdoBtnDetect.rdoBoxSSD.setEnabled(True)
            for v3DEdt in self._guiCtrParams:
                v3DEdt.gui.setEnabled(True)
            self._ckbROI.gui.setEnabled(hasROI)
        self.calibratedThreadsLock.release()

    def disableCalibration(self):
        # Black out frames.
        displayHeight, displayWidth = self._getFrameSize()
        shape = (displayHeight, displayWidth)
        frame = np.ones(shape, np.uint8) # Black image.
        self.updateFrame(frame, self.imgLblLeft, 0, self.txtLblLeft)
        self.updateFrame(frame, self.imgLblRight, 0, self.txtLblRight)

        # Disable access to calibration parameters to prevent thread overflow.
        self.v3DRdoBtnMode.rdoBoxRaw.setEnabled(False)
        self.v3DRdoBtnMode.rdoBoxUnd.setEnabled(False)
        self.v3DRdoBtnMode.rdoBoxStr.setEnabled(False)
        self.v3DRdoBtnDetect.rdoBoxNone.setEnabled(False)
        self.v3DRdoBtnDetect.rdoBoxYOLO.setEnabled(False)
        self.v3DRdoBtnDetect.rdoBoxSSD.setEnabled(False)
        for v3DEdt in self._guiCtrParams:
            v3DEdt.gui.setEnabled(False)
        self._ckbROI.gui.setEnabled(False)

    def closeEvent(self, event):
        # Close application.
        self.signals.stop.emit() # Warn threads to stop.
        self._threadPool.waitForDone() # Wait for threads to stop.
        event.accept()

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
        v3DEdt.gui.returnPressed.connect(v3DEdt.onParameterChanged)
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

    def _createRdoButMode(self, grpBoxLay, param, row, col):
        # Create one parameter.
        lbl = QLabel(param)
        self.v3DRdoBtnMode = Vision3DRadioButtonMode(param, parent=self)
        self.v3DRdoBtnMode.rdoBoxRaw.setChecked(True)
        self.v3DRdoBtnMode.rdoBoxUnd.setChecked(False)
        self.v3DRdoBtnMode.rdoBoxStr.setChecked(False)
        self.v3DRdoBtnMode.rdoBoxRaw.toggled.connect(self.v3DRdoBtnMode.onParameterChanged)
        self.v3DRdoBtnMode.rdoBoxUnd.toggled.connect(self.v3DRdoBtnMode.onParameterChanged)
        self.v3DRdoBtnMode.rdoBoxStr.toggled.connect(self.v3DRdoBtnMode.onParameterChanged)
        grpBoxLay.addWidget(lbl, row+0, col)
        grpBoxLay.addWidget(self.v3DRdoBtnMode.rdoBoxRaw, row+1, col)
        grpBoxLay.addWidget(self.v3DRdoBtnMode.rdoBoxUnd, row+2, col)
        grpBoxLay.addWidget(self.v3DRdoBtnMode.rdoBoxStr, row+3, col)

    def _createRdoButDetection(self, grpBoxLay, param, row, col):
        # Create one parameter.
        lbl = QLabel(param)
        self.v3DRdoBtnDetect = Vision3DRadioButtonDetection(param, parent=self)
        self.v3DRdoBtnDetect.rdoBoxNone.setChecked(True)
        self.v3DRdoBtnDetect.rdoBoxYOLO.setChecked(False)
        self.v3DRdoBtnDetect.rdoBoxSSD.setChecked(False)
        self.v3DRdoBtnDetect.rdoBoxNone.toggled.connect(self.v3DRdoBtnDetect.onParameterChanged)
        self.v3DRdoBtnDetect.rdoBoxYOLO.toggled.connect(self.v3DRdoBtnDetect.onParameterChanged)
        self.v3DRdoBtnDetect.rdoBoxSSD.toggled.connect(self.v3DRdoBtnDetect.onParameterChanged)
        grpBoxLay.addWidget(lbl, row, col+0)
        grpBoxLay.addWidget(self.v3DRdoBtnDetect.rdoBoxNone, row, col+1)
        grpBoxLay.addWidget(self.v3DRdoBtnDetect.rdoBoxYOLO, row, col+2)
        grpBoxLay.addWidget(self.v3DRdoBtnDetect.rdoBoxSSD, row, col+3)

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

    def _convertCvQt(self, frame):
        # Convert frame to pixmap.
        rgbImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        displayHeight, displayWidth, channel = rgbImg.shape
        bytesPerLine = channel * displayWidth
        qtImg = QImage(rgbImg.data, displayWidth, displayHeight, bytesPerLine, QImage.Format_RGB888)
        return QPixmap.fromImage(qtImg)

def cmdLineArgs():
    # Create parser.
    dscr = 'script designed for 3D vision.'
    parser = argparse.ArgumentParser(description=dscr)
    cmdLineArgsVideoStream(parser)
    cmdLineArgsCalibrate(parser, addChessboard=False)
    args = parser.parse_args()

    return vars(args)

# Main program.
if __name__=="__main__":
    # Get command line arguments.
    args = cmdLineArgs()

    # Create Qt application.
    app = QApplication(sys.argv)
    v3D = Vision3D(args)
    v3D.show()
    sys.exit(app.exec_())

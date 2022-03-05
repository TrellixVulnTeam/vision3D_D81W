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
from postThread import PostThread
import cv2
import numpy as np
import threading
from videoStream import cmdLineArgsVideoStream
from calibrate import cmdLineArgsCalibrate
import logging

# Logger.
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
        self.rdoBoxYOLO = QRadioButton('YOLO')
        self.rdoBoxYOLO.mode = 'YOLO'
        self.rdoBoxSSD = QRadioButton('SSD')
        self.rdoBoxSSD.mode = 'SSD'
        self._param = param # Track associated parameter.
        self._vision3D = parent
        grpBtn = QButtonGroup(parent)
        grpBtn.setExclusive(True) # Make radio button exclusive.
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

class Vision3DRadioButtonKeyPoints(QWidget):
    def __init__(self, param, parent=None):
        # Initialise.
        super().__init__(parent)
        self.rdoBoxORB = QRadioButton('ORB')
        self.rdoBoxORB.mode = 'ORB'
        self.rdoBoxSIFT = QRadioButton('SIFT')
        self.rdoBoxSIFT.mode = 'SIFT'
        self._param = param # Track associated parameter.
        self._vision3D = parent
        grpBtn = QButtonGroup(parent)
        grpBtn.setExclusive(True) # Make radio button exclusive.
        grpBtn.addButton(self.rdoBoxORB)
        grpBtn.addButton(self.rdoBoxSIFT)

    def onParameterChanged(self):
        # Callback on parameter change.
        rdoBtn = self.sender()
        if rdoBtn.isChecked():
            # Send signal to threads.
            self._vision3D.disableCalibration()
            value = rdoBtn.mode # Mode which has been modified.
            self._vision3D.signals.changeParam.emit(self._param, 'str', value) # Emit value and associated parameter / type.

class Vision3DRadioButtonSegmentation(QWidget):
    def __init__(self, param, parent=None):
        # Initialise.
        super().__init__(parent)
        self.rdoBoxEnt = QRadioButton('ENet')
        self.rdoBoxEnt.mode = 'ENet'
        self.rdoBoxWsd = QRadioButton('Watershed')
        self.rdoBoxWsd.mode = 'Watershed'
        self.rdoBoxKMs = QRadioButton('KMeans')
        self.rdoBoxKMs.mode = 'KMeans'
        self._param = param # Track associated parameter.
        self._vision3D = parent
        grpBtn = QButtonGroup(parent)
        grpBtn.setExclusive(True) # Make radio button exclusive.
        grpBtn.addButton(self.rdoBoxEnt)
        grpBtn.addButton(self.rdoBoxWsd)
        grpBtn.addButton(self.rdoBoxKMs)

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
        self._args = args.copy()

        # Set up info/debug log on demand.
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%H:%M:%S', level=logging.INFO)

        # Create widgets.
        grpBox = self._createParameters()
        self._imgLblLeft = QLabel()
        self._txtLblLeft = QLabel('Left')
        self._imgLblRight = QLabel()
        self._txtLblRight = QLabel('Right')
        self._imgLblPost = QLabel()
        self._txtLblPost = QLabel('Postprocess')
        self._resetLabels()

        # Handle alignment.
        self._txtLblLeft.setAlignment(Qt.AlignCenter)
        self._imgLblLeft.setAlignment(Qt.AlignCenter)
        self._txtLblRight.setAlignment(Qt.AlignCenter)
        self._imgLblRight.setAlignment(Qt.AlignCenter)
        self._txtLblPost.setAlignment(Qt.AlignCenter)
        self._imgLblPost.setAlignment(Qt.AlignCenter)

        # Handle layout.
        grdLay = QGridLayout()
        grdLay.addWidget(grpBox, 0, 0, 1, 2)
        grdLay.addWidget(self._txtLblLeft, 1, 0)
        grdLay.addWidget(self._txtLblRight, 1, 1)
        grdLay.addWidget(self._imgLblLeft, 2, 0)
        grdLay.addWidget(self._imgLblRight, 2, 1)
        grdLay.addWidget(self._txtLblPost, 3, 0, 1, 2)
        grdLay.addWidget(self._imgLblPost, 4, 0, 1, 2)
        self.setLayout(grdLay)

        # Download DNN pretrained models.
        self._downloadDNNPreTrainedModels()

        # Start threads.
        self._createThreads()

    def updatePrepFrame(self, frame, dct):
        # Update thread image.
        side = dct['side']
        imgLbl = self._imgLblLeft if side == 'left' else self._imgLblRight
        qtImg = self._convertCvQt(frame)
        imgLbl.setPixmap(qtImg)

        # Update thread label.
        txtLbl = self._txtLblLeft if side == 'left' else self._txtLblRight
        txt = txtLbl.text()
        lbl = txt.split()[0] # Suppress old FPS: get first word (title).
        txtLbl.setText(lbl + ' - FPS %d'%dct['fps'])

    def updatePostFrame(self, frame, fmt, msg):
        # Update thread image.
        qtImg = self._convertCvQt(frame, fmt=fmt)
        self._imgLblPost.setPixmap(qtImg)

        # Update thread label.
        txt = self._txtLblPost.text()
        lbl = txt.split()[0] # Suppress old FPS: retrive only first word (Postprocess).
        self._txtLblPost.setText(lbl + ' - ' + msg)

    def calibrationDone(self, vidID, hasROI, params):
        # Re-enable radio buttons when both threads are calibrated.
        self.calibratedThreadsLock.acquire()
        self.calibratedThreads += vidID
        if self.calibratedThreads == self._threadLeft.vidID + self._threadRight.vidID:
            self.calibratedThreads = 0
            self.v3DRdoBtnMode.rdoBoxRaw.setEnabled(True)
            self.v3DRdoBtnMode.rdoBoxUnd.setEnabled(True)
            self.v3DRdoBtnMode.rdoBoxStr.setEnabled(True)
            self.v3DRdoBtnDetect.rdoBoxYOLO.setEnabled(True)
            self.v3DRdoBtnDetect.rdoBoxSSD.setEnabled(True)
            for v3DEdt in self._guiCtrParams:
                v3DEdt.gui.setEnabled(True)
            self._ckbROI.gui.setEnabled(hasROI)
        if 'focXLeft' in params:
            self._focXLeft.gui.setText('%.1f'%params['focXLeft'])
        if 'focXRight' in params:
            self._focXRight.gui.setText('%.1f'%params['focXRight'])
        self.calibratedThreadsLock.release()

    def disableCalibration(self):
        # Disable access to calibration parameters to prevent thread overflow.
        self.v3DRdoBtnMode.rdoBoxRaw.setEnabled(False)
        self.v3DRdoBtnMode.rdoBoxUnd.setEnabled(False)
        self.v3DRdoBtnMode.rdoBoxStr.setEnabled(False)
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

    def _createParameters(self):
        # Create video parameters.
        grpBox = QGroupBox('Parameters')
        grpBoxLay = QGridLayout()
        grpBox.setLayout(grpBoxLay)
        self._guiCtrParams = [] # All GUIs with controls.
        self._createEditParameters(grpBoxLay, 'videoCapWidth', 0, 0)
        self._createEditParameters(grpBoxLay, 'videoCapHeight', 0, 1)
        self._createEditParameters(grpBoxLay, 'videoCapFrameRate', 1, 0)
        if self._args['hardware'] == 'arm-jetson':
            self._createEditParameters(grpBoxLay, 'videoFlipMethod', 1, 1)
            self._createEditParameters(grpBoxLay, 'videoDspWidth', 2, 0)
            self._createEditParameters(grpBoxLay, 'videoDspHeight', 2, 1)
        self._args['focXLeft'] = -1.
        self._focXLeft = self._createEditParameters(grpBoxLay, 'focXLeft', 3, 0, objType='double')
        self._args['focXRight'] = -1.
        self._focXRight = self._createEditParameters(grpBoxLay, 'focXRight', 3, 1, objType='double')

        # Create control parameters.
        self._args['mode'] = 'raw'
        self._createRdoButMode(grpBoxLay, 'mode', 0, 6)
        if self._args['fisheye']:
            tooltip = 'Divisor for new focal length.'
            self._createEditParameters(grpBoxLay, 'fovScale', 2, 7, rowSpanLbl=2, rowSpanGUI=2,
                                       enable=True, objType='double', tooltip=tooltip)
            tooltip = 'Sets the new focal length in range between the min focal length and the max\n'
            tooltip += 'focal length. Balance is in range of [0, 1].'
            self._createEditParameters(grpBoxLay, 'balance', 2, 8, rowSpanLbl=2, rowSpanGUI=2,
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
            self._createEditParameters(grpBoxLay, 'alpha', 2, 8, rowSpanLbl=2, rowSpanGUI=2,
                                       enable=True, objType='double', tooltip=tooltip)
        self._args['ROI'] = False
        self._ckbROI = self._createChkBoxParameters(grpBoxLay, 'ROI', 2, 9, rowSpan=2)
        self._createChkBoxParametersPost(grpBoxLay)
        self._args['DBGcapt'] = False
        self._createChkBoxParameters(grpBoxLay, 'DBGcapt', 1, 28)
        self._args['DBGpost'] = False
        self._createChkBoxParameters(grpBoxLay, 'DBGpost', 2, 28)
        self._args['DBGrun'] = False
        self._createChkBoxParameters(grpBoxLay, 'DBGrun', 1, 29)
        self._args['DBGprof'] = False
        self._createChkBoxParameters(grpBoxLay, 'DBGprof', 2, 29)
        self._args['DBGcomm'] = False
        self._createChkBoxParameters(grpBoxLay, 'DBGcomm', 3, 29)

        # Handle alignment.
        grpBox.setAlignment(Qt.AlignCenter)
        grpBoxLay.setAlignment(Qt.AlignCenter)

        return grpBox

    def _createEditParameters(self, grpBoxLay, param, row, col,
                              rowSpanLbl=1, colSpanLbl=1, rowSpanGUI=1, colSpanGUI=1,
                              enable=False, objType='int', tooltip=None, colOffset=2):
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
        grpBoxLay.addWidget(lbl, row, colOffset*col+0, rowSpanLbl, colSpanLbl)
        grpBoxLay.addWidget(v3DEdt.gui, row, colOffset*col+1, rowSpanGUI, colSpanGUI)
        v3DEdt.gui.setEnabled(enable)
        if enable:
            self._guiCtrParams.append(v3DEdt) # Enabled edits have controls.
        if tooltip:
            lbl.setToolTip(tooltip)

        return v3DEdt

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
        # Create GUI for mode.
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

    def _createChkBoxParametersPostDetection(self, grpBoxLay, row, col):
        # Create GUI for detection.
        self._args['detectMode'] = 'YOLO'
        self.v3DRdoBtnDetect = Vision3DRadioButtonDetection('detectMode', parent=self)
        self.v3DRdoBtnDetect.rdoBoxYOLO.setChecked(True)
        self.v3DRdoBtnDetect.rdoBoxSSD.setChecked(False)
        self.v3DRdoBtnDetect.rdoBoxYOLO.toggled.connect(self.v3DRdoBtnDetect.onParameterChanged)
        self.v3DRdoBtnDetect.rdoBoxSSD.toggled.connect(self.v3DRdoBtnDetect.onParameterChanged)
        grpBoxLay.addWidget(self.v3DRdoBtnDetect.rdoBoxYOLO, row, col+0)
        grpBoxLay.addWidget(self.v3DRdoBtnDetect.rdoBoxSSD, row, col+1)
        self._args['confidence'] = 0.5
        self._createEditParameters(grpBoxLay, 'confidence', row, col+2, enable=True, objType='double')
        self._args['nms'] = 0.3
        self._createEditParameters(grpBoxLay, 'nms', row, col+3, enable=True, objType='double')
        self._args['tracking'] = False
        self._createChkBoxParameters(grpBoxLay, 'tracking', row, col+4)

    def _createChkBoxParametersPostKeypoints(self, grpBoxLay, row, col):
        # Create GUI for keypoints.
        self._args['kptMode'] = 'ORB'
        self.v3DRdoBtnKpt = Vision3DRadioButtonKeyPoints('kptMode', parent=self)
        self.v3DRdoBtnKpt.rdoBoxORB.setChecked(True)
        self.v3DRdoBtnKpt.rdoBoxSIFT.setChecked(False)
        self.v3DRdoBtnKpt.rdoBoxORB.toggled.connect(self.v3DRdoBtnKpt.onParameterChanged)
        self.v3DRdoBtnKpt.rdoBoxSIFT.toggled.connect(self.v3DRdoBtnKpt.onParameterChanged)
        grpBoxLay.addWidget(self.v3DRdoBtnKpt.rdoBoxORB, row, col+0)
        grpBoxLay.addWidget(self.v3DRdoBtnKpt.rdoBoxSIFT, row, col+1)
        self._args['nbFeatures'] = 100
        self._createEditParameters(grpBoxLay, 'nbFeatures', row, col+2, enable=True, objType='int')

    def _createChkBoxParametersSegmentation(self, grpBoxLay, row, col):
        # Create GUI for keypoints.
        self._args['segMode'] = 'Watershed'
        self.v3DRdoBtnSeg = Vision3DRadioButtonSegmentation('segMode', parent=self)
        self.v3DRdoBtnSeg.rdoBoxEnt.setChecked(True)
        self.v3DRdoBtnSeg.rdoBoxWsd.setChecked(True)
        self.v3DRdoBtnSeg.rdoBoxKMs.setChecked(False)
        self.v3DRdoBtnSeg.rdoBoxEnt.toggled.connect(self.v3DRdoBtnSeg.onParameterChanged)
        self.v3DRdoBtnSeg.rdoBoxWsd.toggled.connect(self.v3DRdoBtnSeg.onParameterChanged)
        self.v3DRdoBtnSeg.rdoBoxKMs.toggled.connect(self.v3DRdoBtnSeg.onParameterChanged)
        grpBoxLay.addWidget(self.v3DRdoBtnSeg.rdoBoxEnt, row, col+0)
        grpBoxLay.addWidget(self.v3DRdoBtnSeg.rdoBoxWsd, row, col+1)
        grpBoxLay.addWidget(self.v3DRdoBtnSeg.rdoBoxKMs, row, col+2)
        self._args['K'] = 10
        self._createEditParameters(grpBoxLay, 'K', row, col+2, enable=True, objType='int')
        self._args['attempts'] = 3
        self._createEditParameters(grpBoxLay, 'attempts', row, col+3, enable=True, objType='int')

    def _createChkBoxParametersPost(self, grpBoxLay):
        # Create GUI for postprocessing.
        self._args['detection'] = False
        detectChkBox = self._createChkBoxParameters(grpBoxLay, 'detection', 0, 10)
        self._createChkBoxParametersPostDetection(grpBoxLay, 0, 22)
        self._args['depth'] = False
        depthChkBox = self._createChkBoxParameters(grpBoxLay, 'depth', 1, 10)
        self._args['numDisparities'] = 16
        self._createEditParameters(grpBoxLay, 'numDisparities', 1, 24, enable=True, objType='int')
        self._args['blockSize'] = 15
        self._createEditParameters(grpBoxLay, 'blockSize', 1, 25, enable=True, objType='int')
        self._args['keypoints'] = False
        kptChkBox = self._createChkBoxParameters(grpBoxLay, 'keypoints', 2, 10)
        self._createChkBoxParametersPostKeypoints(grpBoxLay, 2, 22)
        self._args['stitch'] = False
        stitchChkBox = self._createChkBoxParameters(grpBoxLay, 'stitch', 3, 10)
        self._args['crop'] = False
        cropChkBox = self._createChkBoxParameters(grpBoxLay, 'crop', 3, 24)
        self._args['segmentation'] = False
        segChkBox = self._createChkBoxParameters(grpBoxLay, 'segmentation', 4, 10)
        self._createChkBoxParametersSegmentation(grpBoxLay, 4, 22)
        grpBtn = QButtonGroup(self)
        grpBtn.setExclusive(True) # Make radio button exclusive.
        grpBtn.addButton(detectChkBox.gui)
        grpBtn.addButton(depthChkBox.gui)
        grpBtn.addButton(kptChkBox.gui)
        grpBtn.addButton(stitchChkBox.gui)
        grpBtn.addButton(segChkBox.gui)

    @staticmethod
    def _downloadDNNPreTrainedModels():
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

        # Download ENet files.
        if not os.path.isfile('enet-classes.txt'):
            wget.download('https://raw.githubusercontent.com/ishacusp/ARGO_Labs/master/opencv-semantic-segmentation/enet-cityscapes/enet-classes.txt')
        else:
            logger.info('[vision3D] enet-classes.txt has already been downloaded.')
        if not os.path.isfile('enet-colors.txt'):
            wget.download('https://raw.githubusercontent.com/ishacusp/ARGO_Labs/master/opencv-semantic-segmentation/enet-cityscapes/enet-colors.txt')
        else:
            logger.info('[vision3D] enet-colors.txt has already been downloaded.')
        if not os.path.isfile('enet-model.net'):
            wget.download('https://raw.githubusercontent.com/ishacusp/ARGO_Labs/master/opencv-semantic-segmentation/enet-cityscapes/enet-model.net')
        else:
            logger.info('[vision3D] enet-model.net has already been downloaded.')

    def _createThreads(self):
        # Start threads.
        self.signals = Vision3DSignals()
        self._threadPool = QThreadPool() # QThreadPool must be used with QRunnable (NOT QThread).
        self._threadPool.setMaxThreadCount(3)
        videoIDLeft = args['videoIDLeft']
        self._threadLeft = VideoThread(videoIDLeft, self._args, self)
        self._threadLeft.signals.updatePrepFrame.connect(self.updatePrepFrame)
        self._threadLeft.signals.calibrationDone.connect(self.calibrationDone)
        self._threadPool.start(self._threadLeft)
        videoIDRight = args['videoIDRight']
        self._threadRight = VideoThread(videoIDRight, self._args, self)
        self._threadRight.signals.updatePrepFrame.connect(self.updatePrepFrame)
        self._threadRight.signals.calibrationDone.connect(self.calibrationDone)
        self._threadPool.start(self._threadRight)
        self._threadPost = PostThread(self._args, self._threadLeft, self._threadRight, self)
        self._threadPost.signals.updatePostFrame.connect(self.updatePostFrame)
        self._threadPool.start(self._threadPost)

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

    def _resetLabels(self):
        # Resize images.
        displayHeight, displayWidth = self._getFrameSize()
        self._imgLblLeft.resize(displayWidth, displayHeight)
        self._imgLblRight.resize(displayWidth, displayHeight)
        self._imgLblPost.resize(displayWidth, displayHeight)

        # Black out frames.
        shape = (displayHeight, displayWidth)
        frame = np.ones(shape, np.uint8) # Black image.
        self.updatePrepFrame(frame, {'fps': 0, 'side': 'left'})
        self.updatePrepFrame(frame, {'fps': 0, 'side': 'right'})
        self.updatePostFrame(frame, 'GRAY', 'None')

    def _convertCvQt(self, frame, fmt='BGR'):
        # Convert frame to pixmap.
        qtImg = None
        if fmt == 'BGR':
            rgbImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            displayHeight, displayWidth, channel = rgbImg.shape
            bytesPerLine = channel * displayWidth
            qtImg = QImage(rgbImg.data, displayWidth, displayHeight, bytesPerLine, QImage.Format_RGB888)
        elif fmt == 'GRAY':
            displayHeight, displayWidth = frame.shape[0], frame.shape[1]
            channel = 1
            bytesPerLine = channel * displayWidth
            qtImg = QImage(frame, displayWidth, displayHeight, bytesPerLine, QImage.Format_Indexed8)
        assert qtImg is not None, 'unknown CvQt conversion format %s'%fmt

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

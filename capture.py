#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""Handling capture."""

# Imports.
import sys

import argparse
import threading
import cv2

from videoStream import VideoStream, cmdLineArgsVideoStream
from calibrate import cmdLineArgsCalibrate, chessboardCalibration

# Synchronisation barrier.
SYNC = None

# Global variables used as signal events to connect/emit all threads to each others.
SAVE_EVENT = False
QUIT_EVENT = False

def cmdLineArgs():
    """Manage command line arguments."""

    # Create parser.
    dscr = 'script designed to capture frames from video stream for later calibration.'
    parser = argparse.ArgumentParser(description=dscr)
    cmdLineArgsVideoStream(parser, strRightReq=False)
    cmdLineArgsCalibrate(parser)
    parser.add_argument('--startIdx', type=int, default=0, metavar='I',
                        help='define start capture index')
    args = parser.parse_args()

    # Convert calibration parameters.
    args.chessboardX = args.chessboard[0]
    args.chessboardY = args.chessboard[1]
    args.squareSize = args.chessboard[2]

    return vars(args)

class CaptureThread(threading.Thread):
    """Thread handling capture."""

    def __init__(self, args):
        """Initialisation."""

        # Initialise.
        super().__init__()
        self._args = args.copy()
        self._idxFrame = self._args['startIdx']

    def run(self):
        """Run."""

        # Capture frames.
        vid = VideoStream(self._args)
        vidID = self._args['videoID']
        global QUIT_EVENT, SAVE_EVENT
        while vid.isOpened():
            # Get video frame.
            frameOK, frame, _ = vid.read()
            if not frameOK:
                continue

            # Save frame. Show it.
            cv2.imshow(f"stream{vidID:02d} - Video raw [s save, q quit]", frame)

            # Wait for user action.
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): # Press 'q' to quit.
                QUIT_EVENT = True
            if key == ord('s'): # Press 's' to save.
                SAVE_EVENT = True

            # Execute user action.
            if QUIT_EVENT:
                print(f"stream{vidID:02d}: exiting...", flush=True)
                break
            if SAVE_EVENT:
                self.save(frame) # Both threads must take the same picture at the same time.
                SAVE_EVENT = False # All threads reset save flag.
        vid.release()
        cv2.destroyAllWindows()

    def save(self, frame):
        """Save."""

        # Check VISUALLY if chessboards are PROPERLY found on BOTH frames: MANDATORY for correct stereo calibration.
        vidID = self._args['videoID']
        print(f"stream{vidID:02d}: looking for chessboard...", flush=True)
        obj, img = [], []
        ret = chessboardCalibration(self._args, frame, obj, img, msg=f"stream{vidID:02d}:")
        SYNC.wait() # Wait for all chessboards (from all threads) to be checked.

        # Wait to know if chessboard corners are found properly.
        key = None
        if ret:
            while key != 'y' and key != 'n':
                key = input(f"stream{vidID:02d}: keep? [y/n] ")
        else:
            print(f"stream{vidID:02d}: drop", flush=True)
        SYNC.wait() # Wait for all answers (from all threads): keep? drop?

        # Save frame on demand.
        if key == 'y':
            print(f"stream{vidID:02d}: saving frame {self._idxFrame:02d}...", flush=True)
            fileID = f"{self._args['videoType']}{self._args['videoID']}"
            cv2.imwrite(f"{fileID}-{self._idxFrame:02d}.jpg", frame)
            self._idxFrame += 1
        SYNC.wait() # All threads wait for each others.

def main():
    """Main function."""

    # Get command line arguments.
    args = cmdLineArgs()

    # Create threads to handle OpenCV video streams.
    args['videoID'] = args['videoIDLeft']
    mainThd = CaptureThread(args.copy())
    strThd = None
    if args['videoIDRight']:
        args['videoID'] = args['videoIDRight']
        strThd = CaptureThread(args.copy())

    # Create barrier to get threads to do things step-by-step.
    nbThreads = 2 if strThd is not None else 1
    global SYNC
    SYNC = threading.Barrier(nbThreads)

    # Start threads.
    mainThd.start()
    if strThd is not None:
        strThd.start()

    # Wait for threads to be done.
    mainThd.join()
    if strThd is not None:
        strThd.join()

# Main program.
if __name__ == '__main__':
    sys.exit(main())

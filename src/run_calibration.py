#!/usr/bin/env python3
"""Entry point: launch the stereo camera calibration GUI.

Run from anywhere:  python src/run_calibration.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calib.gui import main

if __name__ == "__main__":
    raise SystemExit(main())

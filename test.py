import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot
import argparse
import os
from modules.deblurM1 import *
from modules.evalways import compute_niqe_score
PATH = "/media/allenx570/資料磁碟/disp_data/AGZ_subset/MAV Images/00114.jpg"
IMG_01 = cv2.imread(PATH)






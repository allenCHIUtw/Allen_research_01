import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
 

def vb_blurr_jello(img, vibration_hz, amplitude_pixels, row_readout_time=1e-5):  # vibration blurr

def vb_blurr_harmonic(img, frequency, amplitude, direction_angle, exposure_time=1/500):
    # need to do 

def rotl_blurr_6dof(img, omega_3d, exposure_time, intrinsic_matrix): # rotation movement blurr
   # need intrincix matrix

def lin_blurr_spatial_variant(img, depth_map, velocity_3d ):#linear movement blurr must 6 dof
  # velocity do not means m/s just means the speed of the pixel movemet for generalization


 

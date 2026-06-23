import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot
import argparse
import os
from modules.deblurM1 import *
from modules.evalways import compute_niqe_score

parser = argparse.ArgumentParser()
parser.add_argument("--p", "-path", help="the path of data",type=str,dest="path")
parser.add_argument("--m","-mode", help="the path of data",type=str)
parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="Path to the CSV data file",
    )
args = parser.parse_args()
# PATH = "/media/allenx570/資料磁碟/disp_data/AGZ_subset/MAV Images/00114.jpg"

PATH = args.path
FILEPATH = args.file
IMG_01 = cv2.imread(PATH)
csv_1 = pd.read_csv(FILEPATH)




# dispatch = {
#     'a': function_a,
#     'b': function_b,
#     'c': function_c
# }

# # 3. Execute the selected function
# dispatch[args.mode]()
def main():


if __name__ == "__main__":
    main()


#! /Users/bin/python3.9
# -*- coding: utf-8 -*-
import pydicom
from matplotlib import pyplot as plt
import pickle

d = pydicom.read_file('stage_2_images/ID_fea0567c3.dcm')
plt.imshow(d.pixel_array)

with open("gaze_data/cxr_gaze_data.pkl",mode="rb") as f:
    hoge = pickle.load(f)
    print(hoge.keys())
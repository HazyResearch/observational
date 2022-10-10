# -*- coding: utf-8 -*-

import sys, os
import numpy as np
from utils import load_gaze_data, make_heatmaps
from viz_utils import plot_saccade, plot_heatmap

def main():
    gaze_seqs, labels, img_pths = load_gaze_data(source="cxr",split_type="all",return_img_pths=True)
    heatmaps = make_heatmaps(gaze_seqs,num_patches=8)
    # change data_dir to the directory you have the CXR-P dicom images saved
    data_dir = 'gaze_data/dicom-images-train'
    img_pths = [os.path.join(data_dir,img_pth) for img_pth in img_pths] 
    # choose a random (image,gaze) pair
    # ndx = np.random.randint(len(img_pths))
    file_kikyou = '1.2.276.0.7230010.3.1.4.8323329.4904.1517875185.355709.dcm'
    file_normal ='1.2.276.0.7230010.3.1.4.8323329.4541.1517875183.370160'
    for key in img_pths:
        if file_normal in key:
            print('normal:'+key)
            file_normal = key
        if file_kikyou in key:
            print('kikyou:'+key)
            file_kikyou = key
    ndx = img_pths.index(file_kikyou)
    img_pth, gaze_seq, heatmap = img_pths[ndx], gaze_seqs[ndx], heatmaps[ndx].squeeze()
    plot_heatmap(source="cxr",img_pth=img_pth,heatmap=heatmap)
    plot_saccade(img_pth, gaze_seq, source="cxr")
    ndx = img_pths.index(file_normal)
    img_pth, gaze_seq, heatmap = img_pths[ndx], gaze_seqs[ndx], heatmaps[ndx].squeeze()
    plot_heatmap(source="cxr",img_pth=img_pth,heatmap=heatmap)
    plot_saccade(img_pth, gaze_seq, source="cxr")
    ndx = 1250-1
    img_pth, gaze_seq, heatmap = img_pths[ndx], gaze_seqs[ndx], heatmaps[ndx].squeeze()
    plot_heatmap(source="cxr",img_pth=img_pth,heatmap=heatmap)
    plot_saccade(img_pth, gaze_seq, source="cxr")



if __name__ == '__main__':
    main()

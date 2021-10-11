# Gaze Datasets

We describe our gaze dataset files for the three datasets used in our paper (MICCAI 2021):

Khaled Saab, Sarah Hooper, Nimit Sohoni, Jupinder Parmar, Brian Pogatchnik, Sen Wu, Jared Dunnmon, Hongyang Zhang, Daniel Rubin, and Christopher Ré

[Observational Supervision for Medical Image Classification using Gaze Data](https://web.stanford.edu/~ksaab/media/MICCAI_2021.pdf)

## Datasets

- CXR-P: Detecting pneumothorax from chest X-Rays. Download the dataset from the SIIM-ACR Pneumothorax Segmentation [challenge](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/data).
- CXR-A: Detecting abnormalities from chest X-Rays. Download the dataset from the "Eye Gaze Data for Chest X-rays" paper on [PhysioNet](https://physionet.org/content/egd-cxr/1.0.0/).
- METS: Detecting metastases in brain MRI slices. We are currently working on the necessary PHI protocols within our hospital to make this dataset public.

The two datasets we collected gaze data on are CXR-P and METS. Since METS is an internal PHI dataset, we are unable to release the images at this point. 


## File descriptions 

The pre-processed gaze data is stored as a dictionary named `[dataset_name]_gaze_data.pkl`. 

- The *keys* of the dictionary represent the unique image IDs.
- The *values* of the dictionary represent the pre-processed gaze sequences, `g`. `g` is a list of `(x,y,t)` tuples, where `(x,y)` are the coordinates on the image with values ranging from `[0,1]`, where `(0,0)` represents the top left of the image and `(1,1)` represents the bottom right of the image. `t` represents the time spent fixating on location `(x,y)`. Note: the length of `g` is different for each image, and the raw values of `t` should not be interpreted as seconds (read the pre-processing procedure for more information).


## Data collection 

### hardware and protocol

To collect gaze data, we built custom software to interface with a screen-based [Tobii Pro Nano eye tracker](https://www.tobiipro.com/product-listing/nano/). This state-of-the-art eye tracker is robust to head movements, ethnicity, corrective lenses, and lighting conditions. Each radiologist went through a 9-point calibration process before the start of each session, which was repeated until the instrument reported a high-quality calibration. To control for fatigue, we limited each annotation session to either 25 or 50 images (depending on participant tolerance levels).

While in use, the program displays a single image to the user and collects gaze coordinates. Once the user has analyzed the image, they press a key to indicate the label given to the image. The program then saves the set of gaze coordinates that overlapped with the image and displays the next image.

### pre-processing

We then performed two preprocessing steps: (1) if the eye tracker could not properly detect the participant’s pupils (e.g. due to blinking) then the gaze location was flagged and filtered out, and (2) if a radiologist was not looking at the image we did not collect the gaze locations. Since the eye tracker has a sampling frequency of 60 Hz, the resulting list of fixation coordinates often included `(x,y)` values that were very close to each other. We therefore decided to aggregate coordinates close to each other to form a new `(x,y,t)` tuple using the following code:

```
thresh = 0.05
g = []
x_prev, y_prev = g_raw[0]
t = 1
for (x,y) in g_raw[1:]:
    if abs(x-x_prev) < thresh and abs(y-y_prev) < thresh:
        t += 1
    else:
        g.append((x,y,t))
        x_prev, y_prev = x, y
        t = 1
```
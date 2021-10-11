# Observational Supervision for Medical Image Classification using Gaze Data

This code implements the Gaze-WS and Gaze-MTL methods described in our paper (MICCAI 2021):

Khaled Saab, Sarah Hooper, Nimit Sohoni, Jupinder Parmar, Brian Pogatchnik, Sen Wu, Jared Dunnmon, Hongyang Zhang, Daniel Rubin, and Christopher Ré

[Observational Supervision for Medical Image Classification using Gaze Data](https://web.stanford.edu/~ksaab/media/MICCAI_2021.pdf)

## Abstract

Deep learning models have demonstrated favorable performance on many medical image classification tasks. However, they rely on expensive handlabeled datasets that are time-consuming to create. In this work, we explore a new supervision source to training deep learning models by using gaze data that is passively and cheaply collected during a clinician’s workflow. We focus on three medical imaging tasks, including classifying chest X-ray scans for pneumothorax and brain MRI slices for metastasis, two of which we curated gaze data for. The gaze data consists of a sequence of fixation locations on the image from an expert trying to identify an abnormality. Hence, the gaze data contains rich information about the image that can be used as a powerful supervision source. We first identify a set of gaze features and show that they indeed contain classdiscriminative information. Then, we propose two methods for incorporating gaze features into deep learning pipelines. When no task labels are available, we combine multiple gaze features to extract weak labels and use them as the sole source of supervision (Gaze-WS). When task labels are available, we propose to use the gaze features as auxiliary task labels in a multi-task learning framework (Gaze-MTL). On three medical image classification tasks, our Gaze-WS method without task labels comes within 5 AUROC points (1.7 precision points) of models trained with task labels With task labels, our Gaze-MTL method can improve performance by 2.4 AUROC points (4 precision points) over multiple baselines.

## Setup instructions 

Prerequisites: Make sure you have Python>=3.6 and PyTorch>=1.4 installed. Then, install dependencies with:
```bash
pip install -r requirements.txt
```

Next, either add the base directory of the repository to your PYTHONPATH, or run:
```bash
pip install -e .
```


## Configuration Options

To train a model, run the `train_model.sh` script.

Important configurations:
  - DATA_DIR: directory where the images are saved
  - SOURCE: the dataset, which is one of {cxr, cxr2, mets}, where cxr is CXR-P and cxr2 is CXR-A
  - TASK: the method, which is one of {original, weak_gaze, or gaze-mtl}, where original is Image-Only and weak_gaze is Gaze-WS


## Datasets

We release our gaze data on our two novel datasets for CXR-P and METS.

### Images

- CXR-P: Download the dataset from the SIIM-ACR Pneumothorax Segmentation [challenge](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/data).
- CXR-A: Download the dataset from the "Eye Gaze Data for Chest X-rays" paper on [PhysioNet](https://physionet.org/content/egd-cxr/1.0.0/).
- METS: We are currently working on the necessary PHI protocols within our hospital to make this dataset public.


### Gaze data

Gaze data for the three datasets, with a visualization demo and README, can be found in the gaze_data directory.
<!-- Images can be found on DAWN in the following directories:  
  POET: /dfs/scratch1/ksaab/data/4tb_hdd/POET/shapes  
  CLEVR: /dfs/scratch1/ksaab/data/4tb_hdd/CLEVR/raw_images/images     
  CXR: /dfs/scratch1/ksaab/data/4tb_hdd/CXR/dicom_images    
  Mets: PHI   -->


  
  






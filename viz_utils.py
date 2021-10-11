import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cv2
import pydicom


def plot_heatmap(source, img_pth, heatmap):

    figure, axs = plt.subplots(nrows=1, ncols=2)

    if source == "cxr":
        ds = pydicom.dcmread(img_pth)
        img = ds.pixel_array
    else:
        img = plt.imread(img_pth)

    img_h, img_w = img.shape[0], img.shape[1]

    axs[0].imshow(img, cmap="gray", vmin=0, vmax=255)

    axs[1].imshow(img, cmap="gray", vmin=0, vmax=255)
    hm2 = cv2.resize(heatmap, (img_h, img_w))
    axs[1].imshow(hm2.T, alpha=0.4)

    axs[0].axis("off")
    axs[1].axis("off")
    figure.tight_layout()
    plt.show()


def plot_saccade(img_pth, gaze_seq, source):
    if source == "cxr":
        ds = pydicom.dcmread(img_pth)
        img = ds.pixel_array
    else:
        img = plt.imread(img_pth)
    img_h, img_w = img.shape[0], img.shape[1]

    s = 100
    gaze_x_list = []
    gaze_y_list = []
    size_list = []

    for i, gaze_pt in enumerate(gaze_seq):

        time_spent = gaze_pt[2]

        gaze_x, gaze_y = img_w * gaze_pt[0], img_h * gaze_pt[1]
        gaze_x_list.append(gaze_x)
        gaze_y_list.append(gaze_y)
        size_list.append(s * time_spent)

    if source == "cxr":
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        plt.imshow(img)
    plt.axis("OFF")
    plt.scatter(gaze_x_list, gaze_y_list, s=size_list, alpha=0.7)
    for i in range(len(gaze_x_list)):
        plt.annotate(str(i + 1), (gaze_x_list[i], gaze_y_list[i]))

    plt.show()

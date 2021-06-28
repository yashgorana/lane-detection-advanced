import cv2
import matplotlib.pyplot as plt


def compare_imgs(img1, img2, title1, title2, **kwargs):

    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    plt.figure()
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 9))
    figure.tight_layout()

    ax1.imshow(img1, **kwargs)
    ax1.set_title(title1)

    ax2.imshow(img2, **kwargs)
    ax2.set_title(title2)

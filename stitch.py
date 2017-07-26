import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys


def extract_features(img):
    gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    corners = cv2.goodFeaturesToTrack(gray, 1000, 0.01, 15)
    return [cv2.KeyPoint(c[0, 0], c[0, 1], 2) for c in corners]


def extract_features1(img):
    features = []
    thresh = 0.01 * img.max()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > thresh:
                features.append(cv2.KeyPoint(j, i, 2))
    return features


def show_features(img, features):
    vis = np.zeros_like(img)
    cv2.drawKeypoints(img, features, vis)
    plt.imshow(vis)


def run(img1, img2):
    f1 = extract_features(img1)
    f2 = extract_features(img2)
    show_features(img1, f1)
    plt.show()

if __name__ == '__main__':
    img1 = cv2.imread(sys.argv[1])
    img2 = cv2.imread(sys.argv[2])
    run(img1, img2)

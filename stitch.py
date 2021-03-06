import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


def find_homography(i1, i2):
    feature_detector = cv2.ORB_create()
    gray = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    kp1, d1 = feature_detector.detectAndCompute(gray, None)
    gray = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
    kp2, d2 = feature_detector.detectAndCompute(gray, None)

    feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = feature_matcher.match(d1, d2)
    print("Found %i matches" % len(matches))
    good = sorted(matches, key=lambda x: x.distance)[:int(len(matches) * 0.2)]

    vis = i1.copy()
    vis = cv2.drawMatches(i1, kp1, i2, kp2, good, vis)
    cv2.imwrite("matches.jpg", vis)

    if len(good) > 4:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        H, s = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4)
        return H

    return None


def transform_image_corners(img, H):
    corners = np.array([
        [0, 0, 1],
        [img.shape[1], 0, 1],
        [0, img.shape[0], 1],
        [img.shape[1], img.shape[0], 1]
    ], dtype=np.float64)

    for i in range(corners.shape[0]):
        corners[i] = np.dot(H, corners[i])
        corners[i] /= corners[i, 2]

    return corners


def warp_and_stich(img1, img2, H):
    """
    Warp first image and draw second image over first image. Return stitched image.
    """
    corners = transform_image_corners(img1, H)

    maxx = int(max(corners[0, 0], corners[1, 0], corners[2, 0], corners[3, 0], img2.shape[1]))
    maxy = int(max(corners[0, 1], corners[1, 1], corners[2, 1], corners[3, 1], img2.shape[0]))

    # apply a perspective warp to stitch the images together
    warped = cv2.warpPerspective(img1, H, (maxx, maxy))
    cv2.imwrite("warped.jpg", warped)
    warped[:img2.shape[0], :img2.shape[1]] = img2
    print("Stitched image, shape is", img1.shape)

    return warped


def stitch(images):
    print("Stitching images...")
    stitched = images[0]
    for img in images[1:]:
        H = find_homography(stitched, img)
        np.set_printoptions(precision=3, suppress=True)
        print("Found homography:\n", H)

        # find size of the stitched image
        corners = transform_image_corners(stitched, H)
        inv_h = np.linalg.inv(H)
        corners_inv_h = transform_image_corners(img, inv_h)

        # choose which image to warp, compare out-of-bounds area
        min_h = np.min(corners)
        min_inv_h = np.min(corners_inv_h)
        if min_h > min_inv_h:
            stitched = warp_and_stich(stitched, img, H)
        else:
            invH = np.linalg.inv(H)
            stitched = warp_and_stich(img, stitched, invH)

    return stitched


if __name__ == '__main__':
    images = [cv2.imread(f) for f in sys.argv[1:]]
    img = stitch(images)
    cv2.imwrite("stiched.jpg", img)
    plt.imshow(img)
    plt.show()

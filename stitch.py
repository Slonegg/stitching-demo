import cv2
import numpy as np
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


def stitch(images):
    print("Stitching images...")
    stitched = images[0]
    for img in images[1:]:
        H = find_homography(stitched, img)
        np.set_printoptions(precision=3, suppress=True)
        print("Found homography:\n", H)

        # apply a perspective warp to stitch the images together
        stitched = cv2.warpPerspective(stitched, H, (stitched.shape[1] + img.shape[1], stitched.shape[0] + img.shape[0]))
        cv2.imwrite("t.jpg", stitched)
        stitched[:img.shape[0], :img.shape[1]] = img
        print("Stitched image, shape is", img.shape)

    return stitched


if __name__ == '__main__':
    images = [cv2.imread(f) for f in sys.argv[1:]]
    img = stitch(images)
    cv2.imwrite("stiched.jpg", img)

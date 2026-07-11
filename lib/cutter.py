"""
Antenna current-distribution image processing via PCA + K-Means.

Classes:
    AntennaCutter - PCA decomposition, K-Means thresholding,
                    binarization, and quadrant symmetrization.
"""

import cv2
import numpy as np


class AntennaCutter:
    """Extract binary feature maps from CST current-distribution images."""

    def __init__(self):
        pass

    def PCA_proimg(self, imgp):
        """
        PCA dimensionality reduction on 36 phase-varied images.

        Steps:
          0. Downsample to 200x200, extract the R channel.
          1. Build a super-array: (x*y, 36) from 36 phase images.
          2. SVD decomposition — keep only the 1st principal component.
          3. Reshape back to (x, y).
          4. Normalize to [0, 255] and invert.
          5. Colour-invert (255 - value).

        Args:
            imgp: Path prefix for the 36 image files (img_0.png ... img_35.png).

        Returns:
            2-D uint8 numpy array (x, y) of the processed image.
        """
        # ---- 0. Downsample & extract R channel ----
        img = cv2.imread(imgp + 'img_0.png', 1)
        img = cv2.resize(img, (200, 200))
        _, _, array = cv2.split(img)        # split into B, G, R — keep R
        array = array.T
        x, y = array.shape[0], array.shape[1]

        # ---- 1. Build super-array: each column = one phase ----
        Superarray = np.zeros((x * y, 36), dtype=np.uint8)
        for n in range(36):
            img = cv2.imread(imgp + f'img_{n}.png', 1)
            img = cv2.resize(img, (200, 200))
            _, _, array = cv2.split(img)
            array = array.T
            x, y = array.shape[0], array.shape[1]
            counter = 0
            for i in range(x):
                for j in range(y):
                    Superarray[counter][n] = array[i][j]
                    counter += 1

        # ---- 2. SVD: keep 1st principal component ----
        U, s, VT = np.linalg.svd(Superarray)

        # ---- 3. Reshape back to image dimensions ----
        imgarr = np.zeros((x, y), dtype=np.uint8)
        counter = 0
        for i in range(x):
            for j in range(y):
                # U values are very small — scale up by 10000
                imgarr[i][j] = abs(U[counter][0] * 10000)
                counter += 1

        # ---- 4. Normalize to [0, 255] ----
        min_val = imgarr.min()
        max_val = imgarr.max()
        imgarr = 255 - 255 * ((imgarr - min_val) / (max_val - min_val))

        # ---- 5. Invert colours ----
        for i in range(x):
            for j in range(y):
                imgarr[i][j] = 255 - imgarr[i][j]

        return imgarr

    def Kmeans_getk(self, comparray, kind='radiation'):
        """
        K-Means clustering (k=2) to find the optimal binarization threshold.

        Initial cluster centres: c0=0 (black), c1=255 (white).
        For 'radiation' mode a narrow central strip is excluded from
        clustering (the feed-line region).

        Args:
            comparray: 2-D uint8 image array.
            kind:      'radiation' or 'ground'.

        Returns:
            Scalar threshold k = (c0 + c1) / 2.
        """
        x = comparray.shape[0]
        y = comparray.shape[1]

        c0 = 0       # black
        c1 = 255     # white
        n_iter = 20

        for _ in range(n_iter):
            num_c0, num_c1 = 0, 0
            c0_L, c1_L = 0.0, 0.0

            # -- count per cluster (excluding centre strip for radiation) --
            for i in range(x):
                for j in range(y):
                    if kind == 'radiation':
                        if int(x / 2 - x * 0.04) <= i <= int(x / 2 + x * 0.06):
                            continue
                    if abs(comparray[i, j] - c0) <= abs(comparray[i, j] - c1):
                        num_c0 += 1
                    else:
                        num_c1 += 1

            # -- update cluster centres --
            for i in range(x):
                for j in range(y):
                    if kind == 'radiation':
                        if int(x / 2 - x * 0.04) <= i <= int(x / 2 + x * 0.06):
                            continue
                    if abs(comparray[i, j] - c0) <= abs(comparray[i, j] - c1):
                        c0_L += comparray[i, j] / num_c0
                    else:
                        c1_L += comparray[i, j] / num_c1

            c0 = c0_L
            c1 = c1_L

        return (c0 + c1) / 2

    def img_Binarization(self, comparray_pre, k, kind='radiation'):
        """
        Binarize the image using threshold k.

        For 'radiation': pixels >= k OR within centre strip → 255, else 0.
        For 'ground':    pixels >= k → 255, else 0.

        Args:
            comparray_pre: 2-D uint8 pre-processed image.
            k:             Binarization threshold.
            kind:          'radiation' or 'ground'.

        Returns:
            2-D uint8 binary array.
        """
        x = comparray_pre.shape[0]
        y = comparray_pre.shape[1]
        comparray = np.zeros((x, y), dtype=np.uint8)

        for i in range(x):
            for j in range(y):
                if kind == 'radiation':
                    if (comparray_pre[i, j] >= k or
                            int(x / 2 - x * 0.04) <= i <= int(x / 2 + x * 0.06)):
                        comparray[i, j] = 255
                    else:
                        comparray[i, j] = 0
                elif kind == 'ground':
                    comparray[i, j] = 255 if comparray_pre[i, j] >= k else 0

        return comparray

    def img_Symmetrization(self, comparray_pre):
        """
        Enforce quadrant symmetry using quadrant I as the reference.

        For each quadrant, if a pixel is 255 in the pre-processed image
        at that quadrant's position, its symmetric counterparts in the
        other three quadrants are also set to 255.

        Args:
            comparray_pre: 2-D uint8 binary array.

        Returns:
            2-D uint8 symmetrized binary array.
        """
        x = comparray_pre.shape[0]
        y = comparray_pre.shape[1]
        comparray = np.zeros((x, y), dtype=np.uint8)

        # ---- Quadrant I (bottom-right) → fill all four quadrants ----
        for i in range(int(x / 2), x):
            for j in range(int(y / 2)):
                comparray[i, j] = comparray_pre[i, j]               # Q-I
                comparray[x - i - 1, j] = comparray_pre[i, j]       # Q-II
                comparray[x - i - 1, y - j - 1] = comparray_pre[i, j]  # Q-III
                comparray[i, y - j - 1] = comparray_pre[i, j]       # Q-IV

        # ---- Quadrant II (top-right) ----
        for i in range(int(x / 2)):
            for j in range(int(y / 2)):
                if comparray_pre[i, j] == 255:
                    comparray[i, j] = 255
                    comparray[x - i - 1, j] = 255
                    comparray[i, y - j - 1] = 255
                    comparray[x - i - 1, y - j - 1] = 255

        # ---- Quadrant III (top-left) ----
        for i in range(int(x / 2)):
            for j in range(int(y / 2), y):
                if comparray_pre[i, j] == 255:
                    comparray[i, j] = 255
                    comparray[x - i - 1, y - j - 1] = 255
                    comparray[i, y - j - 1] = 255
                    comparray[x - i - 1, j] = 255

        # ---- Quadrant IV (bottom-left) ----
        for i in range(int(x / 2), x):
            for j in range(int(y / 2), y):
                if comparray_pre[i, j] == 255:
                    comparray[i, j] = 255
                    comparray[i, y - j - 1] = 255
                    comparray[x - i - 1, y - j - 1] = 255
                    comparray[x - i - 1, j] = 255

        return comparray

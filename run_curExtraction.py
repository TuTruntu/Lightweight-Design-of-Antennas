"""
Current Characteristic Extraction
---------------------------------
Extract binary feature maps for the radiation and ground structures
from CST current-distribution images using PCA + K-Means clustering.

Usage:
    python run_extraction.py
"""

import os
import sys

# Ensure the project root is on sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

from lib.cutter import AntennaCutter


# ---- Paths (relative to project root) ----
DATA   = os.path.join(ROOT, 'data')
OUTPUT = os.path.join(ROOT, 'output')

p_p1 = os.path.join(DATA, 'curimg', 'curimg_p_0.5_1.8') + os.sep
p_p2 = os.path.join(DATA, 'curimg', 'curimg_p_0.5_2.3') + os.sep
p_p3 = os.path.join(DATA, 'curimg', 'curimg_p_0.5_2.8') + os.sep

g_p1 = os.path.join(DATA, 'curimg', 'curimg_g_0.4_1.8') + os.sep
g_p2 = os.path.join(DATA, 'curimg', 'curimg_g_0.4_2.3') + os.sep
g_p3 = os.path.join(DATA, 'curimg', 'curimg_g_0.4_2.8') + os.sep

p_chara_name = 'p_chara.npy'
g_chara_name = 'g_chara.npy'


def process(p1, p2, p3, save_name, kind):
    """
    Run PCA + K-Means extraction on three frequency folders,
    merge results via element-wise max, symmetrize, save, and display.

    Args:
        p1, p2, p3:  Paths to image folders for 1.8 / 2.3 / 2.8 GHz.
        save_name:   Output filename (saved under output/charadata/).
        kind:        'radiation' or 'ground'.
    """
    cutter = AntennaCutter()
    chara_dir = os.path.join(OUTPUT, 'charadata')
    save_path = os.path.join(chara_dir, save_name)

    if os.path.exists(save_path):
        print(f'{save_path} already exists — loading cached result.')
        arr_sym = np.load(save_path)
    else:
        prearr1 = cutter.PCA_proimg(p1)
        k1 = cutter.Kmeans_getk(prearr1, kind)
        imgarr1 = cutter.img_Binarization(prearr1, k1, kind)

        prearr2 = cutter.PCA_proimg(p2)
        k2 = cutter.Kmeans_getk(prearr2, kind)
        imgarr2 = cutter.img_Binarization(prearr2, k2, kind)

        prearr3 = cutter.PCA_proimg(p3)
        k3 = cutter.Kmeans_getk(prearr3, kind)
        imgarr3 = cutter.img_Binarization(prearr3, k3, kind)

        # Merge three frequencies via element-wise maximum
        arr = np.maximum(imgarr1, imgarr2, imgarr3)

        # Symmetrize
        arr_sym = cutter.img_Symmetrization(arr)

        os.makedirs(chara_dir, exist_ok=True)
        np.save(save_path, arr_sym)

    # Display result
    image = Image.fromarray(arr_sym.T)
    plt.imshow(image, 'gray')
    plt.show()


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    process(p_p1, p_p2, p_p3, p_chara_name, 'radiation')
    process(g_p1, g_p2, g_p3, g_chara_name, 'ground')

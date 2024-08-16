from __future__ import annotations

import cv2
import numpy as np 
from numpy.typing import NDArray

import matplotlib.pyplot as plt

def viz_slic_superpixel(superpixel_param_list: NDArray, img: NDArray) -> NDArray:

    """
    オリジナルの画像に、superpixelの重心を描画する
    """

    for superpixel_param in superpixel_param_list:
        x, y = superpixel_param[-2:]
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 225), thickness=-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig = plt.figure()
    plt.imshow(img)
    plt.savefig("./init_superpixel_center.png")
from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt

def viz_slic_superpixel(
    superpixel_param_list: NDArray,
    img: NDArray,
    save_file_name: str
) -> None:

    """
    オリジナルの画像に、superpixelの重心を描画する
    """

    for superpixel_param in superpixel_param_list:
        x, y = superpixel_param[-2:]
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 225), thickness=-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # fig = plt.figure()
    # plt.imshow(img)
    cv2.imwrite(f"./{save_file_name}.png", img)

def viz_superpixel_color(
    pixel_label_mat      : NDArray,
    superpixel_param_list: NDArray,
    lab_img              : NDArray
) -> None:

    """
    オリジナルの画像のピクセル値を属するsuperpixelのパラメータの値に変更した結果を可視化
    """

    for superpixel_id, superpixel_param in enumerate(superpixel_param_list):
        idxs = np.where(pixel_label_mat==superpixel_id)
        lab_img[idxs] = superpixel_param[:3]

    rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
    # fig     = plt.figure()
    # plt.imshow(rgb_img)
    cv2.imwrite(f"./slic_result_color.png", rgb_img)
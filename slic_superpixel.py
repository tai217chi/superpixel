
import cv2

import numpy as np

from src.slic.slic import SLIC
from src.slic.util import viz_superpixel_color
from copy import deepcopy

from skimage import io, segmentation, color

def main():
    slic = SLIC(num_superpixel=100, num_iter=10, max_color_dist=10)

    # 画像の読み込み
    img = cv2.imread("./data/len_std.jpg")

    # 初期化プロセス
    slic.init_process(deepcopy(img))

    # クラスタリング
    pixel_label_mat, superpixel_param_list = slic.fit(deepcopy(img))
    lab_img                                = cv2.cvtColor(deepcopy(img), cv2.COLOR_BGR2LAB)
    viz_superpixel_color(pixel_label_mat, superpixel_param_list, lab_img)

    # scikit-image公式実装と比較
    ski_img = io.imread("./data/len_std.jpg")
    label = segmentation.slic(img, compactness=20)
    out = color.label2rgb(label, img, kind = 'avg')
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    io.imsave("lena_skimage.png", out)

if __name__ == "__main__":
    main()
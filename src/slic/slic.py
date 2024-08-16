from __future__ import annotations

import cv2 
import numpy as np 
from numpy.typing import NDArray
from copy import deepcopy
from .util import viz_slic_superpixel

class SLIC:

    _num_superpixel : int
    _num_iter       : int
    _max_color_dist : int

    def __init__(
            self,
            num_superpixel       : int=40,
            num_iter             : int=10,
            max_color_dist       : int = 20 # [1, 40]
    ) -> None:
        
        self._num_superpixel      = num_superpixel
        self._num_iter            = num_iter
        self._max_color_dist      = max_color_dist

    def init_process(
            self, 
            img: NDArray
    ) -> None:
        
        # superpixelを配置する間隔を決める
        num_pixels                = img.shape[0]*img.shape[1]
        self._superpixel_interbal = int(np.sqrt(num_pixels/self._num_superpixel))

        # RGB -> LAB
        img_lab = cv2.cvtColor(deepcopy(img), cv2.COLOR_BGR2Lab)

        # 各ピクセルのパラメータ[l, a, b, x, y]を格納したマトリックスを作成する。
        self._pixel_parameter_mat = self._create_pixel_params(img_lab, img_lab.shape[0], img_lab.shape[1])

        # 各ピクセルのクラスタのインデックスと各ピクセルの最寄りのsuperpixelとの距離関数の値を初期化
        self._pixel_labels     = np.array([-1 for _ in range(img.shape[0]*img.shape[1])]).reshape(img.shape[0], img.shape[1])
        self._pixel_dist_value = np.array([np.inf for _ in range(img.shape[0]*img.shape[1])]).reshape(img.shape[0], img.shape[1])

        # superpixelのパラメータ([l, a, b, x, y]を初期化する)
        self._superpixel_param_list = self._init_superpixel_param(img_lab.shape[0], img_lab.shape[1])

        self._transform_superpixel_center()

        #* デバッグ
        print(f"superpixelの数 (クラスタ数): {self._num_superpixel}")
        print(f"superpixelに統合したときのサイズ: ({self._superpixel_param_list.shape})")
        viz_slic_superpixel(self._superpixel_param_list, img)

    def fit(self):
        # ０．あるsuperpixelから特定の範囲に存在するピクセルのみと距離Dを計算
        # １．各ピクセルのクラスタのインデックスを距離が最も小さいsuperpixelのインデックスに更新する
        # ２．superpixelの重心を計算し直す
        # ３．決められた回数だけ0~2の処理を繰り返す。

        for i in range(self._num_iter):
            pass

    def _init_superpixel_param(
            self, 
            height : int, 
            width  : int
    ) -> NDArray:
        
        """superpixelのパラメータを初期化する
            [l, a, b, x, y]

        Returns:
            NDArray: superpixelの重心位置
        """

        # superpixelのグリッドの行数と列数を決定する
        sup_w, sup_h = int(np.ceil(width / self._superpixel_interbal)), int(np.ceil(height / self._superpixel_interbal))

        # 実際のクラスタ数に更新
        self._num_superpixel      = sup_h * sup_w
        self._superpixel_interbal = int(np.sqrt((width * height) / self._num_superpixel))

        # 等間隔にsuperpixelの座標を決定
        offset_x = int((width - self._superpixel_interbal * (sup_w - 1))/2) # 必ず画像ぴったりにsuperpixelを配置できるとは限らないのでoffsetが必要
        offset_y = int((height - self._superpixel_interbal * (sup_h - 1))/2) # 必ず画像ぴったりにsuperpixelを配置できるとは限らないのでoffsetが必要

        superpixel_params = np.array([], dtype=int)
        for h in range(sup_h):
            for w in range(sup_w):
                y                 = h * self._superpixel_interbal + offset_y
                x                 = w * self._superpixel_interbal + offset_x
                superpixel_params = np.append(superpixel_params, self._pixel_parameter_mat[y, x])

        return superpixel_params.reshape(-1, 5)

    def _transform_superpixel_center(self):

        """
        勾配の情報をもとにクラスタの重心を移動
            もとの重心位置の周辺8近傍の勾配をプレヴィットフィルタを用いて計算する
        """

        for i in range(len(self._superpixel_param_list)):
            base_grad = np.inf
            base_x, base_y = self._superpixel_param_list[i, -2:]

            # 勾配を計算する際の変位量
            dispalacement = [
                (0, 0),
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (-1, -1),
                (1, -1),
                (-1, 1)
            ]

            for d in dispalacement:
                current_x, current_y = int(base_x) + d[0], int(base_y) + d[1]

                # 勾配を計算 (プレヴィットフィルタ)
                vect1 = self._pixel_parameter_mat[current_y, current_x+1, :3] - self._pixel_parameter_mat[current_y, current_x-1, :3]
                vect2 = self._pixel_parameter_mat[current_y+1, current_x, :3] - self._pixel_parameter_mat[current_y-1, current_x, :3]

                gradient = (
                    np.linalg.norm(vect1, ord=2) ** 2 + np.linalg.norm(vect2, ord=2)**2 # 勾配の大きさ
                )

                # 最小な勾配を選択してsuperpixelの重心位置を変更
                if base_grad > gradient:
                    base_grad = gradient
                    base_x    = current_x
                    base_y    = current_y

            # 勾配が最小の位置をsuperpixelの重心位置に設定する
            l, a, b = self._pixel_parameter_mat[base_x, base_y, :3]
            self._superpixel_param_list[i] = [l, a, b, base_x, base_y]

    def _create_pixel_params(
            self, 
            lab_img    : NDArray,
            width      : int, 
            height     : int
    ) -> NDArray :
        
        """
            return:
                parameter matrix: shape(Hight, Width, 5)
        """

        parameter_mat = np.zeros((height, width, 5))

        for y in range(height):
            for x in range(width):
                parameter_mat[y, x, :] = np.array(
                                                    [lab_img[y, x, 0], 
                                                     lab_img[y, x, 1], 
                                                     lab_img[y, x, 2], 
                                                     x               , 
                                                     y               ]
                                                )

        return parameter_mat



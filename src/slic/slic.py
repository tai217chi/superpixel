from __future__ import annotations

import cv2 
import numpy as np 
from numpy.typing import NDArray
from copy import deepcopy
from tqdm import tqdm
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
        self._img_height          = img.shape[0]
        self._img_width           = img.shape[1]
        self._superpixel_interbal = int(np.sqrt(num_pixels/self._num_superpixel))

        # RGB -> LAB
        img_lab = cv2.cvtColor(deepcopy(img), cv2.COLOR_BGR2Lab)

        # 各ピクセルのパラメータ[l, a, b, x, y]を格納したマトリックスを作成する。
        self._pixel_parameter_mat = self._create_pixel_params(img_lab, self._img_height, self._img_width)

        # 各ピクセルのクラスタのインデックスと各ピクセルの最寄りのsuperpixelとの距離関数の値を初期化
        self._pixel_labels     = np.array([-1 for _ in range(num_pixels)]).reshape(self._img_height, self._img_width)
        self._pixel_dist_value = np.array([np.inf for _ in range(num_pixels)]).reshape(self._img_height, self._img_width)

        # superpixelのパラメータ([l, a, b, x, y]を初期化する)
        self._superpixel_param_list = self._init_superpixel_param(self._img_height, self._img_width)

        self._transform_superpixel_center()

        #* デバッグ
        print(f"superpixelの数 (クラスタ数): {self._num_superpixel}")
        print(f"superpixelに統合したときのサイズ: ({self._superpixel_param_list.shape})")
        viz_slic_superpixel(self._superpixel_param_list, img, save_file_name = "initial_superpixel")

    def fit(self, img: NDArray):
        # ０．あるsuperpixelから特定の範囲に存在するピクセルのみと距離Dを計算
        # １．各ピクセルのクラスタのインデックスを距離が最も小さいsuperpixelのインデックスに更新する
        # ２．superpixelの重心を計算し直す
        # ３．決められた回数だけ0~2の処理を繰り返す。

        for i in tqdm(range(self._num_iter)):
            self._update_superpixel()

        viz_slic_superpixel(self._superpixel_param_list, img, "result")


    def _update_superpixel(self):
        
        ## 各ピクセルをsuperpixelに割り当てる処理 ##
        for superpixel_id, superpixel in enumerate(self._superpixel_param_list):

            super_x, suepr_y = int(superpixel[3]), int(superpixel[4])

            x_min = max(0, super_x - self._superpixel_interbal)
            y_min = max(0, suepr_y - self._superpixel_interbal)
            x_max = min(self._img_width - 1, super_x + self._superpixel_interbal)
            y_max = min(self._img_width - 1, self._img_height)

            for y in range(y_min, y_max+1):
                for x in range(x_min, x_max+1):
                    pixel = self._pixel_parameter_mat[y, x]
                    d     = self._calc_distance(superpixel, pixel)

                    if d < self._pixel_dist_value[y, x]:
                        self._pixel_dist_value[y, x] = d
                        self._pixel_labels[y ,x]     = superpixel_id

        ## sueprpixelの重心位置を再計算 -> 同一のsuperpixelに属するpixelのパラメータの平均によって求める ##
        for i in range(self._num_superpixel):
            idxs  = np.where(self._pixel_labels == i) # idxs: ([y1, y2, ...], [x1, x2, ...]), あるsuperpixelに属するすべてのピクセルの座標を取得する
            count = len(idxs[0])
            avg_y = np.round(np.sum(idxs[0]) / count)
            avg_x = np.round(np.sum(idxs[1]) / count)
            avg_l = np.round(np.sum(self._pixel_parameter_mat[idxs][:, 0]) / count)
            avg_a = np.round(np.sum(self._pixel_parameter_mat[idxs][:, 1]) / count)
            avg_b = np.round(np.sum(self._pixel_parameter_mat[idxs][:, 2]) / count)

            self._superpixel_param_list[i] = [avg_l, avg_a, avg_b, avg_x, avg_y]


    def _calc_distance(self, superpixel: NDArray, pixel: NDArray):

        """ピクセルとsuperpixelの距離を計算するための関数

        Args:
            superpoint (NDArray): 特定のsuperpixelのパラメータ, [l, a, d, x, y]
            pixel (_type_): 特定のpixelのパラメータ, [l, a, d, x, y]
        """

        dc = np.sqrt((superpixel[0] - pixel[0])**2 + (superpixel[1] - pixel[1])**2 + (superpixel[2] - pixel[2])*2)
        ds = np.sqrt((superpixel[3] - pixel[3])**2 + (superpixel[4] - pixel[4])**2)
        d  = np.sqrt(dc**2 + (ds**2/self._superpixel_interbal)*self._max_color_dist**2)

        return d

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



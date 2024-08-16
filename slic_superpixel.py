
import cv2

from src.slic.slic import SLIC

def main():
    slic = SLIC(num_superpixel=20, num_iter=10, max_color_dist=20)

    # 画像の読み込み
    img = cv2.imread("./data/len_std.jpg")

    # 初期化プロセス
    slic.init_process(img)

if __name__ == "__main__":
    main()
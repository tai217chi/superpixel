## cudaのイメージをインポート (https://hub.docker.com/r/nvidia/cuda/tags?page=&page_size=&ordering=&name=12.1)
FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04
ARG TORCH=2.2.0

# タイムゾーンの指定。これがないとビルドの途中でCUIインタラクションが発生し停止する
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

## install apt packages.
RUN apt-get update && \
    apt-get install -y \
    wget \
    bzip2 \
    build-essential \
    git \
    git-lfs \
    curl \
    ca-certificates \
    libsndfile1-dev \
    libgl1 \
    python3.8 \
    python3-pip

# キャッシュの削除。これをすることで多少imageが軽くなる。
RUN rm -rf /var/lib/apt/lists/*

# python パッケージのインストール
RUN pip3 install --no-cache-dir matplotlib
RUN pip3 install --no-cache-dir seaborn
RUN pip3 install --no-cache-dir numba
RUN pip3 install --no-cache-dir numpy
RUN pip3 install --no-cache-dir opencv-python
WORKDIR /superpixel
# -*- coding: utf-8 -*-
"""
TIA (Text Image Augmentation) 优化版本
使用 NumPy 向量化操作提升性能

实现了基于移动最小二乘法（Moving Least Squares, MLS）的图像变形算法，
用于 OCR 文本识别中的图像增强，包括扭曲、拉伸和透视变换。
"""

import numpy as np


class WarpMLS:
    """
    基于移动最小二乘法（MLS）的图像变形类

    使用 MLS 算法实现图像从源控制点到目标控制点的变形，
    支持多种图像变形效果，如扭曲、拉伸和透视变换。
    """

    def __init__(self, src, src_pts, dst_pts, dst_w, dst_h, trans_ratio=1.0):
        """
        初始化 WarpMLS 对象

        参数:
            src (numpy.ndarray): 源图像，格式为 (H, W, C) 或 (H, W)
            src_pts (list): 源控制点列表，格式为 [[x1, y1], [x2, y2], ...]
            dst_pts (list): 目标控制点列表，格式为 [[x1, y1], [x2, y2], ...]
            dst_w (int): 目标图像宽度
            dst_h (int): 目标图像高度
            trans_ratio (float, optional): 变换强度，默认为 1.0
        """
        self.src = src
        self.src_pts = np.array(src_pts, dtype=np.float32)
        self.dst_pts = np.array(dst_pts, dtype=np.float32)
        self.pt_count = len(self.dst_pts)
        self.dst_w = dst_w
        self.dst_h = dst_h
        self.trans_ratio = trans_ratio
        self.grid_size = 100  # 网格大小，控制变形精度
        self.rdx = np.zeros((self.dst_h, self.dst_w), dtype=np.float32)  # x方向位移场
        self.rdy = np.zeros((self.dst_h, self.dst_w), dtype=np.float32)  # y方向位移场

    @staticmethod
    def __bilinear_interp(x, y, v11, v12, v21, v22):
        """
        双线性插值函数

        参数:
            x (numpy.ndarray): x方向插值系数，范围 [0, 1]
            y (numpy.ndarray): y方向插值系数，范围 [0, 1]
            v11: 左上角像素值
            v12: 右上角像素值
            v21: 左下角像素值
            v22: 右下角像素值

        返回:
            numpy.ndarray: 插值结果
        """
        return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 * (1 - y) + v22 * y) * x

    def generate(self):
        """
        生成变形图像

        步骤:
            1. 计算位移场 (calc_delta)
            2. 根据位移场生成变形图像 (gen_img)

        返回:
            numpy.ndarray: 变形后的图像，格式与输入图像相同
        """
        self.calc_delta()
        return self.gen_img()

    def calc_delta(self):
        """
        计算位移场

        为每个网格点计算从源图像到目标图像的位移量，
        存储在 self.rdx 和 self.rdy 中。
        """
        if self.pt_count < 2:
            return

        # 生成网格点坐标
        grid_indices_i = np.arange(0, self.dst_w, self.grid_size)
        grid_indices_j = np.arange(0, self.dst_h, self.grid_size)

        for i in grid_indices_i:
            for j in grid_indices_j:
                # 当前网格点坐标
                cur_pt = np.array([i, j], dtype=np.float32)

                # 向量化计算所有控制点的权重
                diff = self.dst_pts - cur_pt
                dist_sq = np.sum(diff ** 2, axis=1)

                # 避免除零
                dist_sq[dist_sq == 0] = np.finfo(np.float32).eps
                w = 1.0 / dist_sq

                # 检查是否在控制点上
                on_control_point = np.any(dist_sq < 0.5)

                if on_control_point:
                    # 在控制点上，直接使用对应的源点
                    k = np.argmin(dist_sq)
                    new_pt = self.src_pts[k]
                else:
                    # MLS 变换计算
                    sw = np.sum(w)
                    swp = np.sum(w[:, np.newaxis] * self.dst_pts, axis=0)
                    swq = np.sum(w[:, np.newaxis] * self.src_pts, axis=0)

                    pstar = swp / sw
                    qstar = swq / sw

                    # 计算局部仿射变换
                    pt_i = self.dst_pts - pstar
                    miu_s = np.sum(w * np.sum(pt_i ** 2, axis=1))

                    cur_pt_shifted = cur_pt - pstar
                    cur_pt_j = np.array([-cur_pt_shifted[1], cur_pt_shifted[0]], dtype=np.float32)

                    # 向量化计算仿射变换
                    pt_j = np.column_stack([-pt_i[:, 1], pt_i[:, 0]])

                    dot_pi_cur = np.sum(pt_i * cur_pt_shifted, axis=1)
                    dot_pj_cur = np.sum(pt_j * cur_pt_shifted, axis=1)
                    dot_pi_curj = np.sum(pt_i * cur_pt_j, axis=1)
                    dot_pj_curj = np.sum(pt_j * cur_pt_j, axis=1)

                    tmp_pt = np.zeros((self.pt_count, 2), dtype=np.float32)
                    tmp_pt[:, 0] = (dot_pi_cur * self.src_pts[:, 0] - dot_pj_cur * self.src_pts[:, 1]) * w / miu_s
                    tmp_pt[:, 1] = (-dot_pi_curj * self.src_pts[:, 0] + dot_pj_curj * self.src_pts[:, 1]) * w / miu_s

                    new_pt = np.sum(tmp_pt, axis=0) + qstar

                # 存储位移量
                self.rdx[j, i] = new_pt[0] - i
                self.rdy[j, i] = new_pt[1] - j

    def gen_img(self):
        """
        根据位移场生成变形图像

        使用网格化方式处理图像，对每个网格块进行双线性插值，
        从源图像采样生成目标图像。

        返回:
            numpy.ndarray: 变形后的图像
        """
        src_h, src_w = self.src.shape[:2]
        dst = np.zeros_like(self.src, dtype=np.float32)

        # 生成所有网格块的坐标
        grid_i = np.arange(0, self.dst_h, self.grid_size)
        grid_j = np.arange(0, self.dst_w, self.grid_size)

        for i in grid_i:
            for j in grid_j:
                ni = min(i + self.grid_size, self.dst_h - 1)
                nj = min(j + self.grid_size, self.dst_w - 1)
                h = ni - i + 1
                w = nj - j + 1

                # 向量化生成网格坐标
                di = np.arange(h, dtype=np.float32).reshape(-1, 1)
                dj = np.arange(w, dtype=np.float32).reshape(1, -1)

                # 归一化坐标
                di_norm = di / h
                dj_norm = dj / w

                # 双线性插值计算位移
                delta_x = self.__bilinear_interp(
                    di_norm, dj_norm,
                    self.rdx[i, j], self.rdx[i, nj],
                    self.rdx[ni, j], self.rdx[ni, nj]
                )
                delta_y = self.__bilinear_interp(
                    di_norm, dj_norm,
                    self.rdy[i, j], self.rdy[i, nj],
                    self.rdy[ni, j], self.rdy[ni, nj]
                )

                # 计算新坐标
                nx = j + dj + delta_x * self.trans_ratio
                ny = i + di + delta_y * self.trans_ratio

                # 边界裁剪
                nx = np.clip(nx, 0, src_w - 1)
                ny = np.clip(ny, 0, src_h - 1)

                nxi = np.floor(nx).astype(np.int32)
                nyi = np.floor(ny).astype(np.int32)
                nxi1 = np.ceil(nx).astype(np.int32)
                nyi1 = np.ceil(ny).astype(np.int32)

                # 双线性插值采样
                if len(self.src.shape) == 3:
                    x = np.tile((ny - nyi)[:, :, np.newaxis], (1, 1, 3))
                    y = np.tile((nx - nxi)[:, :, np.newaxis], (1, 1, 3))
                else:
                    x = ny - nyi
                    y = nx - nxi

                dst[i:i+h, j:j+w] = self.__bilinear_interp(
                    x, y,
                    self.src[nyi, nxi], self.src[nyi, nxi1],
                    self.src[nyi1, nxi], self.src[nyi1, nxi1]
                )

        dst = np.clip(dst, 0, 255).astype(np.uint8)
        return dst


def tia_distort(src, segment=4):
    """
    文本图像扭曲变换

    对输入图像应用随机扭曲变换，模拟自然场景中的文本变形，
    用于增强 OCR 模型对变形文本的识别能力。

    参数:
        src (numpy.ndarray): 输入图像，格式为 (H, W, C) 或 (H, W)
        segment (int, optional): 图像宽度方向的分段数，默认为 4

    返回:
        numpy.ndarray: 扭曲后的图像
    """
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut // 3

    src_pts = [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]
    dst_pts = [
        [np.random.randint(thresh), np.random.randint(thresh)],
        [img_w - np.random.randint(thresh), np.random.randint(thresh)],
        [img_w - np.random.randint(thresh), img_h - np.random.randint(thresh)],
        [np.random.randint(thresh), img_h - np.random.randint(thresh)]
    ]

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        src_pts.extend([[cut * cut_idx, 0], [cut * cut_idx, img_h]])
        dst_pts.extend([
            [cut * cut_idx + np.random.randint(thresh) - half_thresh, np.random.randint(thresh) - half_thresh],
            [cut * cut_idx + np.random.randint(thresh) - half_thresh, img_h + np.random.randint(thresh) - half_thresh]
        ])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    return trans.generate()


def tia_stretch(src, segment=4):
    """
    文本图像拉伸变换

    对输入图像应用随机拉伸变换，模拟不同角度拍摄的文本，
    用于增强 OCR 模型对拉伸文本的识别能力。

    参数:
        src (numpy.ndarray): 输入图像，格式为 (H, W, C) 或 (H, W)
        segment (int, optional): 图像宽度方向的分段数，默认为 4

    返回:
        numpy.ndarray: 拉伸后的图像
    """
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut * 4 // 5

    src_pts = [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]
    dst_pts = [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        move = np.random.randint(thresh) - half_thresh
        src_pts.extend([[cut * cut_idx, 0], [cut * cut_idx, img_h]])
        dst_pts.extend([[cut * cut_idx + move, 0], [cut * cut_idx + move, img_h]])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    return trans.generate()


def tia_perspective(src):
    """
    文本图像透视变换

    对输入图像应用随机透视变换，模拟不同视角拍摄的文本，
    用于增强 OCR 模型对透视变形文本的识别能力。

    参数:
        src (numpy.ndarray): 输入图像，格式为 (H, W, C) 或 (H, W)

    返回:
        numpy.ndarray: 透视变换后的图像
    """
    img_h, img_w = src.shape[:2]
    thresh = img_h // 2

    src_pts = [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]
    dst_pts = [
        [0, np.random.randint(thresh)],
        [img_w, np.random.randint(thresh)],
        [img_w, img_h - np.random.randint(thresh)],
        [0, img_h - np.random.randint(thresh)]
    ]

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    return trans.generate()

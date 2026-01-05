# coding: utf-8
import random
import logging
import traceback
from typing import Union

import cv2
import torch
import numpy as np
import albumentations as alb
from albumentations.core.transforms_interface import ImageOnlyTransform
from tia import tia_distort, tia_stretch, tia_perspective

logger = logging.getLogger(__name__)


def RandomSkewAug(angle=[0, 0], p=0.5):
    """保持图片大小的倾斜变换"""
    if isinstance(angle, (list, tuple)) and len(angle) == 2:
        shear_range = {"x": (angle[0], angle[1])}
    else:
        shear_range = {"x": (-angle, angle)}
    return alb.Affine(shear=shear_range, p=p)

def RandomTwitsAug(scale=100, offset=0, p=0.5):
    """保持图片大小的扭曲变换"""
    grid_size = max(2, min(8, scale // 20))
    magnitude = min(50, scale // 2)
    return alb.GridElasticDeform(num_grid_xy=(grid_size, grid_size), magnitude=magnitude, p=p)

class CustomRandomCrop(ImageOnlyTransform):
    """
    自定义随机裁剪变换类 - 从图像的四个边缘随机裁剪指定大小

    该变换通过从图像的上、下、左、右四个边缘随机裁剪来模拟图像边界被截断的效果，
    有助于增强模型对不完整文本图像的识别能力。

    原理说明：
    - 从图像的四个边缘分别随机裁剪指定大小的区域
    - 确保裁剪后的图像尺寸不小于原始尺寸的一定比例
    - 将裁剪后的图像resize回原始尺寸

    参数说明：
        crop_size (tuple): 裁剪大小，格式为(height, width)，表示从每个边缘裁剪的最大尺寸
        p (float, 可选): 应用此变换的概率，默认为1.0
    """

    def __init__(self, crop_size, p=1.0):
        """
        初始化CustomRandomCrop变换实例

        参数：
            crop_size: 裁剪大小，(height, width)格式
            p: 变换应用概率
        """
        super(CustomRandomCrop, self).__init__(p)
        self.crop_size = crop_size

    def cal_params(self, img):
        """
        计算裁剪参数

        该函数尝试在10次内找到合适的裁剪参数，确保裁剪后的图像
        尺寸不小于原始尺寸的50%（或最小4像素）。

        参数：
            img: 输入图像

        返回：
            tuple: (h_top, w_left, h, w)
                - h_top: 顶部裁剪尺寸
                - w_left: 左侧裁剪尺寸
                - h: 裁剪后的高度
                - w: 裁剪后的宽度
        """
        ori_h, ori_w = img.shape[:2]

        # 尝试10次找到合适的裁剪参数
        for _ in range(10):
            # 随机生成四个边缘的裁剪尺寸
            h_top, h_bot = (
                random.randint(0, self.crop_size[0]),
                random.randint(0, self.crop_size[0]),
            )
            w_left, w_right = (
                random.randint(0, self.crop_size[1]),
                random.randint(0, self.crop_size[1]),
            )

            # 计算裁剪后的尺寸
            h = ori_h - h_top - h_bot
            w = ori_w - w_left - w_right

            # 检查裁剪后的尺寸是否满足最小要求（不小于原始尺寸的50%或最小4像素）
            if h < max(ori_h * 0.5, 4) or w < max(ori_w * 0.5, 4):
                continue  # 不满足要求，继续尝试

            return h_top, w_left, h, w

        # 如果10次都没找到合适参数，返回不裁剪的结果
        return 0, 0, ori_h, ori_w

    def apply(self, img, **params):
        """
        对输入图像应用随机裁剪变换

        参数：
            img: 输入图像
            **params: 其他参数（由albumentations框架传入）

        返回：
            裁剪并resize后的图像，尺寸与输入图像相同
        """
        # 计算裁剪参数
        h_top, w_left, h, w = self.cal_params(img)

        # 只有在实际需要裁剪时才进行resize
        if h_top != 0 or w_left != 0 or h != img.shape[0] or w != img.shape[1]:
            # 执行裁剪并resize回原始尺寸
            out = cv2.resize(
                img[h_top : h_top + h, w_left : w_left + w], img.shape[:2][::-1]
            )

            # 确保输出图像的维度与输入一致
            if img.ndim > out.ndim:
                out = np.expand_dims(out, axis=-1)

            return out

        # 无需变换时直接返回原图，避免不必要的resize操作
        return img


class TransparentOverlay(ImageOnlyTransform):
    """模仿标注笔的标注效果。"""

    def __init__(
        self, max_height_ratio, max_width_ratio, alpha, p=1.0
    ):
        super(TransparentOverlay, self).__init__(p)
        self.max_height_ratio = max_height_ratio
        self.max_width_ratio = max_width_ratio
        self.alpha = alpha

    def apply(self, img, x=0, y=0, height=0, width=0, color=(0, 0, 0), **params):
        if min(height, width) < 2:
            return img
        original_c = img.shape[2]

        # 确保图片有四个通道（RGBA）
        if img.shape[2] < 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        # 创建一个与图片大小相同的覆盖层
        overlay = img.copy()

        # 在覆盖层上涂色
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)

        # 结合覆盖层和原图片
        img = cv2.addWeighted(overlay, self.alpha, img, 1 - self.alpha, 0)

        # Convert the image back to the original number of channels
        if original_c != img.shape[2]:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img

    @property
    def targets_as_params(self):
        return ['image']

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width, _ = img.shape

        # Compute the actual pixel values for the maximum height and width
        max_height = int(height * self.max_height_ratio)
        max_width = int(width * self.max_width_ratio)

        x = np.random.randint(0, max(width - max_width, 1))
        y = np.random.randint(0, max(height - max_height, 1))
        rect_width = np.random.randint(0, max(max_width, 1))
        rect_height = np.random.randint(0, max(max_height, 1))

        color = [np.random.randint(0, 256) for _ in range(3)]

        return {
            'x': x,
            'y': y,
            'width': rect_width,
            'height': rect_height,
            'color': color,
        }


class TiaDistort(ImageOnlyTransform):
    """TIA - 应用文本图像增强技术 - 扭曲变换"""

    def apply(self, img, **params):
        """
        对输入图像应用TIA变换

        参数：
            img: 输入图像
            **params: 其他参数（由albumentations框架传入）

        返回：
            应用TIA变换后的图像
        """
        img_height, img_width = img.shape[0:2]

        if img_height >= 32 and img_width >= 32:
            # 扭曲变换（需要足够大的图像）
            return tia_distort(img, segment=10)

        return img

class TiaStretch(ImageOnlyTransform):
    """TIA - 应用文本图像增强技术 - 拉伸变换"""

    def apply(self, img, **params):
        """
        对输入图像应用TIA变换

        参数：
            img: 输入图像
            **params: 其他参数（由albumentations框架传入）

        返回：
            应用TIA变换后的图像
        """
        img_height, img_width = img.shape[0:2]

        # 根据随机数选择对应的TIA变换
        if img_height >= 32 and img_width >= 32:
            # 拉伸变换（需要足够大的图像）
            return tia_stretch(img, 10)

        return img
class TiaPerspective(ImageOnlyTransform):
    """TIA - 应用文本图像增强技术 - 透视变换"""

    def apply(self, img, **params):
        """
        对输入图像应用TIA变换

        参数：
            img: 输入图像
            **params: 其他参数（由albumentations框架传入）

        返回：
            应用TIA变换后的图像
        """

        return tia_perspective(img)

class TransformWrapper(object):
    """
    变换包装器类 - 将albumentations变换转换为torchvision风格的变换

    该类作为桥梁，将albumentations库的图像变换包装成可以直接用于
    PyTorch模型的变换，处理数据类型转换和维度调整。

    原理说明：
    - 支持torch.Tensor和numpy.ndarray两种输入类型
    - 自动处理异常，确保变换失败时返回有效结果
    - 将HWC格式转换为CHW格式，符合PyTorch标准

    参数说明：
        transform (alb.Compose): albumentations变换组合
    """

    def __init__(self, transform: alb.Compose):
        """
        初始化TransformWrapper实例

        参数：
            transform: albumentations变换组合
        """
        self.transform = transform  # 保存变换组合

    def __call__(self, ori_image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        将albumentations的transform转换成torchvision的transform

        该函数处理输入图像的格式转换，应用变换，并确保输出符合PyTorch标准格式

        参数：
            ori_image: 输入图像，可以是torch.Tensor（[H, W, C]）或numpy.ndarray

        返回：
            torch.Tensor: 变换后的图像，形状为[C, H, W]
        """
        # 转换输入为numpy数组
        if isinstance(ori_image, torch.Tensor):
            image = ori_image.numpy()
        else:
            image = ori_image

        try:
            # 应用albumentations变换
            out = self.transform(image=image)['image']
        except Exception as e:
            # 变换失败时记录错误并返回原始图像
            logger.error(f"Error when transforming one image with shape: {image.shape}, {e}")
            traceback.print_exc()
            # 确保返回torch.Tensor且为float32类型
            if isinstance(ori_image, torch.Tensor):
                return ori_image.to(torch.float32)
            else:
                return torch.from_numpy(ori_image.astype(np.float32))

        # 确保输出有正确的维度
        if image.ndim > out.ndim:
            out = np.expand_dims(out, axis=-1)

        # 转换HWC格式为CHW格式，并转换为torch.Tensor
        out = torch.from_numpy(out.transpose((2, 0, 1)))  # to: [C, H, W]
        return out


# 预定义的变换组合配置
# 这些变换组合针对不同的训练阶段和任务类型进行了优化

# 文本识别训练阶段的变换组合
# 主要包含几何变形和TIA增强，用于提高模型对变形文本的鲁棒性
rec_train_transform = TransformWrapper(alb.Compose(
    [
        # 注释掉的变换可以根据需要启用
        # alb.OneOf([Erosion((2, 3)), Dilation((2, 3))], p=0.1),  # 形态学操作
        # alb.RandomBrightnessContrast(0.1, 0.1, True, p=0.1),  # 亮度对比度调整
        # alb.GaussianBlur((3, 3), p=0.2),  # 高斯模糊
        # alb.Emboss(p=0.3, alpha=(0.2, 0.5), strength=(0.2, 0.7)),  # 浮雕效果
        # alb.OpticalDistortion(...),  # 光学畸变
        # alb.Sharpen(p=0.3, alpha=(0.2, 0.5), lightness=(0.5, 1.0)),  # 锐化

        # 随机几何变形
        alb.OneOf([
            RandomSkewAug(angle = [-25, 0], p=1.0),  # 随机倾斜（向左倾斜0-25度）
            RandomTwitsAug(scale = 5.0, offset = 0, p=1.0), # 随机扭曲
            TiaDistort(p=1.0),# 扭曲变换
            TiaStretch(p=1.0),# 拉伸变换
            TiaPerspective(p=1.0),# 透视变换
            ],
            p=0.3,
        ),

        # 颜色处理（当前被注释掉）
        # alb.InvertImg(p=0.3),  # 颜色反转
        # ToSingleChannelGray(p=1.0),  # 转灰度图
        # CustomNormalize(p=1.0),  # 自定义归一化

        # 标准化（使用预计算的均值和标准差）
        alb.Normalize(0.456045, 0.224567),
    ]
))

# 文本识别微调阶段的变换组合
# 包含更多的图像质量增强，模拟真实场景的图像退化
rec_ft_transform = TransformWrapper(alb.Compose(
    [
        # 形态学操作：腐蚀或膨胀（二选一，10%概率）
        alb.OneOf([alb.Morphological(scale=(2, 3), operation="erosion", p=0.5), alb.Morphological(scale=(2, 3), operation="dilation", p=0.5)], p=0.1),

        # TransparentOverlay(1.0, 0.1, alpha=0.4, p=0.2),  # 半透明覆盖层

        # 图像质量增强
        alb.RandomBrightnessContrast(0.1, 0.1, True, p=0.1),  # 亮度对比度调整
        # alb.ImageCompression(95, p=0.3),  # 图像压缩
        # alb.GaussNoise(20, p=0.2),  # 高斯噪声
        alb.GaussianBlur((3, 3), p=0.1),  # 高斯模糊

        # 纹理增强
        alb.Emboss(p=0.3, alpha=(0.2, 0.5), strength=(0.2, 0.7)),  # 浮雕效果

        # 光学畸变（相机模式，20%概率）
        alb.OpticalDistortion(
            p = 0.2,
            distort_limit = (-0.05, 0.05),  # 畸变限制
            interpolation = 0,  # 插值方式
            mode = "camera"  # 相机模式
        ),

        # alb.Affine(scale={"x": (0.8, 1.2), "y": (1.0, 1.0)}, p=0.2),  # 随机拉伸

        alb.GaussianBlur((3, 3), p=0.1),  # 再次高斯模糊
        alb.InvertImg(p=0.3),  # 颜色反转

        # 转换为灰度图并归一化
        alb.ToGray(num_output_channels=1, method="weighted_average", p=1.0),
        # CustomNormalize(p=1.0),  # 自定义归一化
        alb.Normalize(0.456045, 0.224567),  # 标准化
    ]
))

# 文本检测训练阶段的变换组合
# 相对轻量的增强，主要用于检测任务
det_train_transform = TransformWrapper(alb.Compose(
    [
        # 注释掉的变换可以根据需要启用
        # alb.OneOf([Erosion((2, 3)), Dilation((2, 3))], p=0.1),  # 形态学操作
        # alb.RandomBrightnessContrast(0.1, 0.1, True, p=0.1),  # 亮度对比度

        # 轻量的图像质量增强
        alb.GaussianBlur((3, 3), p=0.1),  # 高斯模糊
        alb.Emboss(p=0.1, alpha=(0.2, 0.5), strength=(0.2, 0.7)),  # 浮雕效果

        # 注释掉的重度畸变变换
        # alb.OpticalDistortion(...),  # 光学畸变
        # alb.Sharpen(...),  # 锐化
        # alb.GaussNoise(10, p=0.2),  # 高斯噪声
        # alb.InvertImg(p=0.1),  # 颜色反转

        # 基础预处理：灰度化 + 归一化
        alb.ToGray(num_output_channels=1, method="weighted_average", p=1.0),
        alb.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0, p=1.0),
        # alb.Normalize(0.456045, 0.224567),  # 可选的标准化
    ]
))

# 文本检测微调阶段的变换组合
# 包含更多的增强，用于提高检测模型的鲁棒性
det_ft_transform = TransformWrapper(alb.Compose(
    [
        # 形态学操作：腐蚀或膨胀（二选一，10%概率）
        alb.OneOf([alb.Morphological(scale=(2, 3), operation="erosion", p=1.0), alb.Morphological(scale=(2, 3), operation="dilation", p=1.0)], p=0.1),

        # TransparentOverlay(1.0, 0.1, alpha=0.4, p=0.2),  # 半透明覆盖层

        # 图像质量增强
        alb.RandomBrightnessContrast(0.1, 0.1, True, p=0.1),  # 亮度对比度
        # alb.ImageCompression(95, p=0.3),  # 图像压缩
        # alb.GaussNoise(20, p=0.2),  # 高斯噪声
        alb.GaussianBlur((3, 3), p=0.1),  # 高斯模糊

        # 纹理增强
        alb.Emboss(p=0.3, alpha=(0.2, 0.5), strength=(0.2, 0.7)),  # 浮雕效果

        # 光学畸变（相机模式，20%概率）
        alb.OpticalDistortion(
            p = 0.2,
            distort_limit = (-0.05, 0.05),  # 畸变限制
            interpolation = 0,  # 插值方式
            mode = "camera"  # 相机模式
        ),

        # alb.Affine(scale={"x": (0.8, 1.2), "y": (1.0, 1.0)}, p=0.2),  # 随机拉伸

        alb.GaussianBlur((3, 3), p=0.1),  # 再次高斯模糊
        alb.InvertImg(p=0.3),  # 颜色反转

        # 转换为灰度图并归一化
        alb.ToGray(num_output_channels=1, method="weighted_average", p=1.0),
        alb.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0, p=1.0),
        # alb.Normalize(0.456045, 0.224567),  # 可选的标准化
    ]
))

# 测试阶段的变换组合
# 最简化的变换，只进行必要的预处理
test_transform = TransformWrapper(alb.Compose(
    [
        # 基础预处理：灰度化 + 归一化
        alb.ToGray(num_output_channels=1, method="weighted_average", p=1.0),  # 转换为灰度图
        alb.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0, p=1.0),  # 归一化到0-1范围
        # alb.Normalize(0.456045, 0.224567),  # 可选的标准化
    ]
))

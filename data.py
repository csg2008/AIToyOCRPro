import random
import re
import string
from typing import List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from transforms import rec_train_transform

# 字符集：数字+大小写字母+特殊字符
BLANK='<blank>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
OTHER_PAD = [BLANK, SOS_TOKEN, EOS_TOKEN]
VOCAB = list(string.digits) + list(string.ascii_letters) + list(string.punctuation) + [' '] + OTHER_PAD
VOCAB_SIZE = len(VOCAB)
OTHER_PAD_SIZE = len(OTHER_PAD)
char2idx = {char: idx for idx, char in enumerate(VOCAB)}
idx2char = {idx: char for char, idx in char2idx.items()}
blank_id   = char2idx[BLANK]
sos_id     = char2idx[SOS_TOKEN]
eos_id     = char2idx[EOS_TOKEN]

# 形近字权重表（优化版）
CONFUSE_WEIGHT_OPTIMIZED = {
    # 字母数字 - 易混淆字符
    '0': 2.0, 'O': 2.0, 'o': 1.8,
    '1': 2.0, 'l': 1.8, 'I': 1.8, '|': 1.8, 'i': 1.5,
    '2': 1.5, 'Z': 1.3, 'z': 1.3,
    '5': 1.5, 'S': 1.3, 's': 1.3,
    '6': 1.5, 'b': 1.3,
    '8': 1.5, 'B': 1.3,

    # 字母 - 易混淆字符
    'p': 1.3, 'q': 1.3,
    'u': 1.2, 'v': 1.2, 'n': 1.2,
    'c': 1.2, 'e': 1.2,

    # 标点符号
    ',': 1.5, '.': 1.5, ';': 1.3, ':': 1.3,
    "'": 1.4, '"': 1.4, '`': 1.3,
    '-': 1.2, '_': 1.2, '—': 1.2,
    '(': 1.3, ')': 1.3, '[': 1.3, ']': 1.3,

    # 空格
    ' ': 1.1,
}

class RandomStringGenerator:
    """
    随机字符串生成器
    1. 字符只能从 VOCAB 列表取
    2. 长度在 [min_chars, max_chars] 之间随机
    3. 按规则随机插入非连续空格
    """

    def __init__(self,
                 vocab: List[str],
                 min_chars: int = 10,
                 max_chars: int = 50):
        """
        :param vocab: 可用字符池（List[str]）
        :param min_chars: 生成字符串的最小长度（不含空格）
        :param max_chars: 生成字符串的最大长度（不含空格）
        """
        if not vocab:
            raise ValueError("VOCAB 不能为空")
        if min_chars < 1 or max_chars < min_chars:
            raise ValueError("min_chars 必须 ≥1 且 ≤ max_chars")

        self.vocab = vocab
        self.min_chars = min_chars
        self.max_chars = max_chars

    def _insert_spaces(self, chars: List[str]) -> List[str]:
        """
        在 s 中随机插入非连续空格。
        规则：每 5~10 个字符可插入一个空格，且不会出现连续空格。
        """
        if len(chars) <= self.min_chars:
            return chars

        # 候选插入位置：索引 5~len(s)-1 处（保证前面至少有 5 个字符）
        # 为了“非连续”，我们记录上一个空格插入的位置
        last_space_pos = -2          # 初始化为 -2，保证第一次不会冲突
        step_range = range(6, 15)    # 6~15 的步长

        pos = 5
        while pos < len(chars):
            # 如果上一个空格紧挨着我，就跳过
            if pos - last_space_pos == 1:
                pos += 1
                continue

            # 30% 概率，随机决定是否在此插入空格
            if random.random() < 0.3:
                chars.insert(pos, ' ')
                last_space_pos = pos
                # 插入后，后面的字符整体后移，直接跳到下一个候选段
                pos += random.choice(step_range)
            else:
                pos += 1

        return chars

    def generate(self) -> str:
        """主入口：生成符合要求的随机字符串"""

        length = random.randint(self.min_chars, self.max_chars)
        raw = random.choices(self.vocab, k=length)
        raw = self._insert_spaces(raw)

        return re.sub(r'\s+', ' ', ''.join(raw)).strip()

class OCRDataset(Dataset):
    ''' 随机合成文本图片数据集 '''

    def __init__(self, num_samples: int, img_height: int, min_width: int, min_chars: int, max_chars: int):
        self.num_samples = num_samples
        self.img_height = img_height
        self.min_width = min_width
        self.set_chars_range(min_chars, max_chars)

        try:
            self.font = ImageFont.truetype("arial.ttf", 24)
        except OSError:
            print("无法加载 'arial.ttf' 字体，将使用系统默认字体。")
            self.font = ImageFont.load_default()

    def set_chars_range(self, min_chars: int, max_chars: int):
        '''设置生成的字符数范围'''
        self.gen = RandomStringGenerator(
                    vocab=VOCAB[:-OTHER_PAD_SIZE],
                    min_chars=min_chars,
                    max_chars=max_chars
                )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 随机生成文字
        text = self.gen.generate()

        # 计算文本像素宽度（用 getbbox 最准确）
        bbox = self.font.getbbox(text)                # (left, top, right, bottom)
        text_w = bbox[2] - bbox[0]                    # 文字宽度
        pad_w = random.randint(8, 20)             # 左右边距 8 ~ 20 像素
        img_w = text_w + pad_w                        # 真实所需宽度
        width = max(img_w, self.min_width)            # 不低于全局最小

        # 生成随机背景颜色（RGB）
        bg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # 文字颜色为背景颜色的反色
        text_color = (255 - bg_color[0], 255 - bg_color[1], 255 - bg_color[2])

        # 创建RGB图像
        img = Image.new('RGB', (width, self.img_height), bg_color)
        draw = ImageDraw.Draw(img)

        # 绘制文字
        left = random.randint(3, width - text_w - 3)
        draw.text((left, 3), text, font=self.font, fill=text_color)

        # 文字转索引
        text_indices = [char2idx[c] for c in text]
        text_tensor = torch.tensor(text_indices, dtype=torch.long)

        return {
            'image': np.array(img),
            'text': text_tensor,
            'text_length': len(text),
            'width': width
        }

def collate_fn(batch):
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    text_lengths = [item['text_length'] for item in batch]
    widths = [item['width'] for item in batch]

    # ✅ 将宽度向上对齐到 16 的倍数对齐（ViT patch）
    def round_up_to_patch_size(w, patch_size=16):
        return ((w + patch_size - 1) // patch_size) * patch_size

    max_width = round_up_to_patch_size(max(widths))

    bs = len(images)
    height = images[0].shape[0]
    images_tensor = torch.ones(bs, 3, height, max_width)
    patch = []
    for i, im in enumerate(images):
        im = rec_train_transform(im)
        w = im.shape[2]
        images_tensor[i, :, :, :w] = im
        patch.append(max(w // 16, 1))                   # patch 长度
    labels = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=blank_id)
    label_lens = torch.tensor(text_lengths, dtype=torch.long)

    # 创建有效宽度掩码
    mask = torch.zeros(len(widths), max_width, dtype=torch.bool)
    for i, w in enumerate(widths):
        mask[i, :w] = True

    # 处理文字序列（不变）
    text_lengths = [len(text) for text in texts]
    max_text_len = max(text_lengths)
    padded_texts = []
    for text in texts:
        padding = max_text_len - len(text)
        text = torch.cat([torch.tensor([sos_id]), text, torch.tensor([eos_id])])
        padded = torch.nn.functional.pad(text, (0, padding), value=blank_id)
        padded_texts.append(padded)
    texts_tensor = torch.stack(padded_texts)

    return {
        'mask': mask,
        'images': images_tensor,
        'patch': torch.tensor(patch),
        'labels': labels,
        'label_lens': label_lens,
        'labels_ce': texts_tensor,
        'text_lengths': torch.tensor(text_lengths),
        'widths': torch.tensor([max_width] * len(batch))  # 统一宽度
    }

if __name__ == "__main__":
    full_set = OCRDataset(num_samples=10,
                        img_height=32,
                        min_width=128,
                        min_chars=10,
                        max_chars=50)
    data = [
        full_set[i] for i in range(1)
    ]

    batch = collate_fn(data)
    print(batch['labels'].shape, batch['labels'])
    print(batch['labels_ce'].shape, batch['labels_ce'])

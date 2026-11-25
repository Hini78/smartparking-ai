"""CRNN для чтения российских автомобильных номеров (SmartParking AI, часть 3).

Использование (нужны только torch, numpy, pillow):

    from crnn_ocr import CRNNReader
    reader = CRNNReader("crnn_best.pt")      # device выбирается сам (cuda, если есть)
    reader.read("crop.png")                  # -> "A123BC77"
    reader.read_with_conf("crop.png")        # -> ("A123BC77", 0.98)
    reader.read_batch([img1, img2, ...])     # -> ["A123BC77", "K211PA69", ...]

На вход идёт УЖЕ вырезанный номер (путь к файлу, PIL.Image или numpy-массив).
Искать табличку на всей фотографии не нужно: это делает детектор из части 2.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ALPHABET = "0123456789ABEKMHOPCTYX"   # 22 символа
BLANK = 0                             # индекс blank для CTC
IMG_H, IMG_W = 32, 160

CHAR2IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}   # 0 занят под blank
IDX2CHAR = {i + 1: c for i, c in enumerate(ALPHABET)}
NUM_CLASSES = len(ALPHABET) + 1


# --------------------------------------------------------------------------
# Предобработка: серое изображение, пропорции сохраняем, дополняем до 32x160
# --------------------------------------------------------------------------
def to_gray_pil(img, bgr: bool = False) -> Image.Image:
    """Принимает путь, PIL.Image или numpy-массив (H,W) / (H,W,3) / (H,W,4)."""
    if isinstance(img, (str, bytes)) or hasattr(img, "__fspath__"):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, :3]
            if bgr:                      # кроп из cv2 имеет порядок BGR
                arr = arr[:, :, ::-1]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return img.convert("L")


def preprocess(img, h: int = IMG_H, w: int = IMG_W, bgr: bool = False) -> np.ndarray:
    """-> uint8 массив (h, w): масштаб по высоте/ширине без искажений, справа padding."""
    g = to_gray_pil(img, bgr=bgr)
    ow, oh = g.size
    scale = min(h / oh, w / ow)
    nw, nh = max(1, round(ow * scale)), max(1, round(oh * scale))
    g = g.resize((nw, nh), Image.BILINEAR)
    arr = np.asarray(g, dtype=np.uint8)
    pad_val = int(np.median(arr))        # фон добавляем в цвет самого кропа
    canvas = np.full((h, w), pad_val, dtype=np.uint8)
    top = (h - nh) // 2
    canvas[top:top + nh, :nw] = arr
    return canvas


def to_tensor(arr_u8: np.ndarray) -> torch.Tensor:
    """uint8 (H,W) или (B,H,W) -> float32 (B,1,H,W) в диапазоне [-1, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(arr_u8)).float() / 255.0
    if x.ndim == 2:
        x = x.unsqueeze(0)
    return ((x - 0.5) / 0.5).unsqueeze(1)


# --------------------------------------------------------------------------
# Модель: CNN -> BiLSTM -> Linear (-> CTC)
# --------------------------------------------------------------------------
def _conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class CRNN(nn.Module):
    """Вход (B,1,32,160) -> логиты (B,T=40,num_classes)."""

    def __init__(self, num_classes: int = NUM_CLASSES, rnn_hidden: int = 192,
                 rnn_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.cnn = nn.Sequential(
            _conv(1, 32), nn.MaxPool2d(2, 2),                       # 16 x 80
            _conv(32, 64), nn.MaxPool2d(2, 2),                      # 8 x 40
            _conv(64, 128), _conv(128, 128), nn.MaxPool2d((2, 1)),  # 4 x 40
            _conv(128, 256), _conv(256, 256), nn.MaxPool2d((2, 1)), # 2 x 40
        )
        self.rnn = nn.LSTM(256 * 2, rnn_hidden, num_layers=rnn_layers,
                           bidirectional=True, batch_first=True,
                           dropout=dropout if rnn_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        f = self.cnn(x)                                   # B,C,H,W
        b, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, w, c * h)    # B,T,C*H
        out, _ = self.rnn(f)
        return self.fc(self.drop(out))                    # B,T,classes


# --------------------------------------------------------------------------
# Кодирование / декодирование CTC
# --------------------------------------------------------------------------
def encode_text(text: str) -> list[int]:
    return [CHAR2IDX[c] for c in text]


def ctc_greedy_decode(logits: torch.Tensor):
    """logits (B,T,C) -> (список строк, список уверенностей)."""
    probs = logits.float().softmax(-1)
    conf, idx = probs.max(-1)                             # B,T
    idx, conf = idx.cpu().numpy(), conf.cpu().numpy()
    texts, confs = [], []
    for seq, cf in zip(idx, conf):
        chars, cs, prev = [], [], BLANK
        for k, p in zip(seq, cf):
            if k != prev and k != BLANK:
                chars.append(IDX2CHAR[int(k)])
                cs.append(float(p))
            prev = k
        texts.append("".join(chars))
        confs.append(float(np.mean(cs)) if cs else 0.0)
    return texts, confs


# --------------------------------------------------------------------------
# Метрики
# --------------------------------------------------------------------------
def edit_ops(ref: str, hyp: str):
    """Левенштейн с разбором: (замены S, пропуски D, лишние I)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            S += int(ref[i - 1] != hyp[j - 1]); i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return S, D, I


def compute_metrics(refs, hyps) -> dict:
    """CER, точность символов, Exact Match. Одни и те же для всех моделей."""
    N = S = D = I = em = 0
    for r, h in zip(refs, hyps):
        s, d, i = edit_ops(r, h)
        S += s; D += d; I += i; N += len(r)
        em += int(r == h)
    return {
        "cer": (S + D + I) / max(N, 1),            # лишние + пропущенные + заменённые
        "char_acc": (N - S - D) / max(N, 1),       # доля символов эталона, прочитанных верно
        "exact_match": em / max(len(refs), 1),     # номер целиком совпал
        "n": len(refs), "S": S, "D": D, "I": I,
    }


# --------------------------------------------------------------------------
# Удобная обёртка для использования весов другим человеком / частью 4
# --------------------------------------------------------------------------
class CRNNReader:
    def __init__(self, weights_path: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.alphabet = ckpt.get("alphabet", ALPHABET)
        assert self.alphabet == ALPHABET, "алфавит весов не совпадает с crnn_ocr.ALPHABET"
        self.h, self.w = ckpt.get("img_h", IMG_H), ckpt.get("img_w", IMG_W)
        self.model = CRNN(**ckpt.get("model_cfg", {}))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()
        self.info = {k: v for k, v in ckpt.items() if k not in ("model_state",)}

    @torch.no_grad()
    def read_batch_with_conf(self, imgs, bgr: bool = False, batch_size: int = 128):
        texts, confs = [], []
        for i in range(0, len(imgs), batch_size):
            arr = np.stack([preprocess(im, self.h, self.w, bgr=bgr) for im in imgs[i:i + batch_size]])
            logits = self.model(to_tensor(arr).to(self.device))
            t, c = ctc_greedy_decode(logits)
            texts += t; confs += c
        return texts, confs

    def read_batch(self, imgs, bgr: bool = False):
        return self.read_batch_with_conf(imgs, bgr=bgr)[0]

    def read_with_conf(self, img, bgr: bool = False):
        t, c = self.read_batch_with_conf([img], bgr=bgr)
        return t[0], c[0]

    def read(self, img, bgr: bool = False) -> str:
        return self.read_with_conf(img, bgr=bgr)[0]

"""Часть 4, ядро демо: фотография или кадр -> решение о допуске.

Порядок работы (из 04_demo.md):
    1. детектор YOLO11n (weights/n640_best.pt) ищет номер на всём кадре;
       если рамок несколько — берём рамку с наибольшей уверенностью;
    2. вырезаем рамку с запасом 5-10% с каждой стороны, без поворота и отражения;
    3. CRNN (weights/crnn_best.pt) читает кроп: строка латиницей + уверенность 0..1;
    4. строка проверяется по формату российского номера;
    5. сравнение с белым списком -> одно из трёх решений.

Решения:
    MANUAL  (ручная проверка) — номер не найден, низкая уверенность или формат не прошёл;
    ALLOW   (пустить)          — формат прошёл, уверенность достаточная, номер в белом списке;
    DENY    (отказать)         — формат прошёл, уверенность достаточная, номера в списке нет.

Проверка без сайта (одна фотография -> рамка, текст, уверенность, решение, время):

    python demo_core.py фото.jpg --whitelist A123BC777
    python demo_core.py фото.jpg --whitelist A123BC777,K211PA69 --conf 0.85 --margin 0.07
"""
from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from crnn_ocr import CRNNReader

# -------------------------------------------------------------------------- #
# Константы
# -------------------------------------------------------------------------- #
DETECTOR_WEIGHTS = "weights/n640_best.pt"   # YOLO11n, вход 640, mAP@50 0.984
READER_WEIGHTS = "weights/crnn_best.pt"    # CRNN, Exact Match 98.9% на val

PLATE_RE = re.compile(r"^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$")

DEFAULT_CONF_THRESHOLD = 0.85   # стартовый порог из 04_demo.md
DEFAULT_MARGIN = 0.07           # запас вокруг рамки: 5-10% с каждой стороны
VIDEO_MAX_FRAMES = 12           # сколько кадров читаем из видео для голосования

DECISION_ALLOW = "ALLOW"
DECISION_DENY = "DENY"
DECISION_MANUAL = "MANUAL"

DECISION_LABELS = {
    DECISION_ALLOW: "ПУСТИТЬ",
    DECISION_DENY: "ОТКАЗАТЬ",
    DECISION_MANUAL: "РУЧНАЯ ПРОВЕРКА",
}

# Русские буквы, похожие на латинские, -> латиница (тот же алфавит, что у CRNN)
CYR_TO_LAT = str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")


def normalize_plate(s: str) -> str:
    """Приводит строку к виду, который возвращает CRNN (латиница, без пробелов)."""
    return s.upper().translate(CYR_TO_LAT).replace(" ", "").replace("\t", "")


# -------------------------------------------------------------------------- #
# Результаты
# -------------------------------------------------------------------------- #
@dataclass
class FrameResult:
    """Результат обработки одного кадра (фотографии)."""
    box: tuple[float, float, float, float] | None  # x1, y1, x2, y2 в пикселях
    det_conf: float | None                         # уверенность детектора
    text: str                                      # что прочитала CRNN (может быть пустым)
    ocr_conf: float                                # уверенность CRNN 0..1
    format_ok: bool
    decision: str                                  # ALLOW / DENY / MANUAL
    reason: str
    elapsed_ms: float
    crop: np.ndarray | None = None                 # кроп, который уходил в CRNN


@dataclass
class VideoResult:
    """Результат обработки видео: голосование по нескольким кадрам."""
    text: str
    ocr_conf: float                  # средняя уверенность кадров, проголосовавших за текст
    votes: int                       # сколько кадров проголосовало за этот текст
    frames_read: int                 # сколько кадров вообще дали чтение
    decision: str
    reason: str
    elapsed_ms: float
    readings: list[dict]             # все чтения: text, ocr_conf, det_conf
    best: FrameResult | None         # кадр с этим текстом для показа на экране
    best_frame: np.ndarray | None = None  # полный кадр, соответствующий best


# -------------------------------------------------------------------------- #
# Решение по одной строке
# -------------------------------------------------------------------------- #
def decide(text: str, ocr_conf: float, whitelist: set[str],
           conf_threshold: float = DEFAULT_CONF_THRESHOLD) -> tuple[str, str]:
    """Правило из 04_demo.md. Возвращает (решение, причина)."""
    if text and PLATE_RE.fullmatch(text):
        if ocr_conf >= conf_threshold:
            if text in whitelist:
                return DECISION_ALLOW, "формат прошёл, номер есть в белом списке"
            return DECISION_DENY, "формат прошёл, номера нет в белом списке"
        return DECISION_MANUAL, (
            f"уверенность OCR {ocr_conf:.2f} ниже порога {conf_threshold:.2f}")
    return DECISION_MANUAL, (
        f"строка «{text or '—'}» не проходит формат российского номера"
        if text else "номер на кадре не найден")


# -------------------------------------------------------------------------- #
# Пайплайн
# -------------------------------------------------------------------------- #
def crop_with_margin(frame: np.ndarray, box, margin: float = DEFAULT_MARGIN) -> np.ndarray:
    """Вырезает рамку с запасом `margin` (доля стороны) с каждой стороны.

    Кроп не поворачивается и не отражается: CRNN училась на горизонтальных
    номерах, которые занимают почти весь кроп.
    """
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    dx, dy = bw * margin, bh * margin
    xa = max(0, int(round(x1 - dx)))
    ya = max(0, int(round(y1 - dy)))
    xb = min(w_img, int(round(x2 + dx)))
    yb = min(h_img, int(round(y2 + dy)))
    if xb - xa < 2 or yb - ya < 2:               # защита от вырожденной рамки
        xa, ya, xb, yb = 0, 0, w_img, h_img
    return frame[ya:yb, xa:xb]


class PlatePipeline:
    """Детектор + CRNN + правило решения. Один экземпляр на процесс."""

    def __init__(self, detector_weights: str = DETECTOR_WEIGHTS,
                 reader_weights: str = READER_WEIGHTS):
        from ultralytics import YOLO          # импорт здесь: тяжёлые зависимости грузятся один раз
        self.detector = YOLO(detector_weights)
        self.reader = CRNNReader(reader_weights)

    # -- шаг 1: найти номер на всём кадре --------------------------------- #
    def detect_box(self, frame_bgr: np.ndarray):
        """Лучший по уверенности бокс детектора -> (box, det_conf) или (None, None)."""
        pred = self.detector.predict(frame_bgr, imgsz=640, verbose=False)[0]
        if pred.boxes is None or len(pred.boxes) == 0:
            return None, None
        i = int(pred.boxes.conf.argmax())
        box = tuple(float(v) for v in pred.boxes.xyxy[i])
        return box, float(pred.boxes.conf[i])

    # -- шаги 1-6: один кадр -> результат --------------------------------- #
    def process_frame(self, frame_bgr: np.ndarray, whitelist: set[str],
                      conf_threshold: float = DEFAULT_CONF_THRESHOLD,
                      margin: float = DEFAULT_MARGIN) -> FrameResult:
        t0 = time.perf_counter()
        box, det_conf = self.detect_box(frame_bgr)
        if box is None:
            decision, reason = DECISION_MANUAL, "детектор не нашёл номер на кадре"
            return FrameResult(box=None, det_conf=None, text="", ocr_conf=0.0,
                               format_ok=False, decision=decision, reason=reason,
                               elapsed_ms=(time.perf_counter() - t0) * 1000)

        crop = crop_with_margin(frame_bgr, box, margin)
        text, ocr_conf = self.reader.read_with_conf(crop, bgr=True)
        format_ok = bool(text) and PLATE_RE.fullmatch(text) is not None
        decision, reason = decide(text, ocr_conf, whitelist, conf_threshold)
        return FrameResult(box=box, det_conf=det_conf, text=text,
                           ocr_conf=float(ocr_conf), format_ok=format_ok,
                           decision=decision, reason=reason,
                           elapsed_ms=(time.perf_counter() - t0) * 1000,
                           crop=crop)

    # -- видео: несколько чтений одной машины -> голосование --------------- #
    def process_video(self, video_path: str, whitelist: set[str],
                      conf_threshold: float = DEFAULT_CONF_THRESHOLD,
                      margin: float = DEFAULT_MARGIN,
                      max_frames: int = VIDEO_MAX_FRAMES,
                      progress=None) -> VideoResult:
        """Читает до `max_frames` равномерно распределённых кадров видео.

        Текст результата — тот, у которого сумма уверенностей больше.
        Уверенность результата — средняя по кадрам, проголосовавшим за этот текст.
        Дальше то же правило, что для фотографии.
        """
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"не удалось открыть видео {video_path}")
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:                    # поток без метаданных
                total = 1_000_000
            step = max(1, total // max_frames)
            indices = list(range(0, total, step))[:max_frames]

            readings: list[dict] = []
            best_by_det: dict[str, FrameResult] = {}
            best_frames: dict[str, np.ndarray] = {}   # полный кадр для показа на сайте
            done = 0
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                res = self.process_frame(frame, whitelist, conf_threshold, margin)
                done += 1
                if res.box is not None and res.text:
                    readings.append({"text": res.text, "ocr_conf": res.ocr_conf,
                                     "det_conf": res.det_conf})
                    prev = best_by_det.get(res.text)
                    if prev is None or (res.det_conf or 0) > (prev.det_conf or 0):
                        best_by_det[res.text] = res
                        best_frames[res.text] = frame
                if progress is not None:
                    progress(min(done, len(indices)) / len(indices))
        finally:
            cap.release()

        if not readings:
            return VideoResult(text="", ocr_conf=0.0, votes=0, frames_read=0,
                               decision=DECISION_MANUAL,
                               reason="ни в одном кадре не удалось прочитать номер",
                               elapsed_ms=(time.perf_counter() - t0) * 1000,
                               readings=[], best=None, best_frame=None)

        # голосование: сумма уверенностей за каждый текст
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for r in readings:
            sums[r["text"]] += r["ocr_conf"]
            counts[r["text"]] += 1
        best_text = max(sums, key=sums.get)
        best_conf = sums[best_text] / counts[best_text]
        decision, reason = decide(best_text, best_conf, whitelist, conf_threshold)
        return VideoResult(text=best_text, ocr_conf=best_conf,
                           votes=counts[best_text], frames_read=len(readings),
                           decision=decision, reason=reason,
                           elapsed_ms=(time.perf_counter() - t0) * 1000,
                           readings=readings, best=best_by_det.get(best_text),
                           best_frame=best_frames.get(best_text))


# -------------------------------------------------------------------------- #
# Проверка без сайта: python demo_core.py фото.jpg ...
# -------------------------------------------------------------------------- #
def _draw_box(frame_bgr: np.ndarray, box) -> np.ndarray:
    out = frame_bgr.copy()
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
    cv2.putText(out, "plate", (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="фотография -> рамка, текст, решение, время")
    ap.add_argument("image", help="путь к фотографии (кадр целиком, не кроп)")
    ap.add_argument("--whitelist", default="",
                    help="номера через запятую, например A123BC777,K211PA69")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD,
                    help="порог уверенности OCR (по умолчанию 0.85)")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                    help="запас вокруг рамки, доля стороны (по умолчанию 0.07)")
    ap.add_argument("--save-debug", default="", help="куда сохранить кадр с рамкой")
    args = ap.parse_args()

    whitelist = {normalize_plate(p) for p in args.whitelist.split(",") if p.strip()}
    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"не удалось открыть {args.image}")

    pipe = PlatePipeline()
    res = pipe.process_frame(frame, whitelist, args.conf, args.margin)

    print(f"рамка        : {None if res.box is None else tuple(round(v, 1) for v in res.box)}")
    print(f"детектор conf: {None if res.det_conf is None else round(res.det_conf, 3)}")
    print(f"текст        : {res.text or '—'}")
    print(f"OCR conf     : {res.ocr_conf:.3f}")
    print(f"формат       : {'ок' if res.format_ok else 'не прошёл'}")
    print(f"решение      : {DECISION_LABELS[res.decision]} ({res.reason})")
    print(f"время        : {res.elapsed_ms:.0f} мс")

    if res.box is not None and args.save_debug:
        cv2.imwrite(args.save_debug, _draw_box(frame, res.box))
        print(f"кадр с рамкой: {args.save_debug}")


if __name__ == "__main__":
    main()

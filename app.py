"""Часть 4. Сайт-демонстрация SmartParking AI — консоль оператора паркинга.

Запуск из корня репозитория:

    streamlit run app.py

Интерфейс как у реального сервиса: экран решения на въезде (пустить /
отказать / ручная проверка), индикация шлагбаума, резиденты (белый список),
журнал въездов со статистикой за день. Самим шлагбаумом сайт не управляет —
на защите достаточно решения на экране.

Веса: детектор weights/n640_best.pt (YOLO11n, вход 640),
чтение weights/crnn_best.pt (CRNN, Exact Match 98,9% на val).
"""
from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import access_db
from demo_core import (
    DECISION_ALLOW,
    DECISION_DENY,
    DECISION_LABELS,
    DECISION_MANUAL,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_MARGIN,
    PLATE_RE,
    VIDEO_MAX_FRAMES,
    PlatePipeline,
    normalize_plate,
)

st.set_page_config(page_title="SmartParking — въезд", page_icon="🅿️",
                   layout="wide", initial_sidebar_state="collapsed")

DB_PATH = Path("data/processed/access.sqlite")

PHOTO_EXTS = ("png", "jpg", "jpeg", "webp", "bmp")
VIDEO_EXTS = ("mp4", "avi", "mov", "mkv")

CAMERAS = ("Въезд №1 — барьер A", "Въезд №2 — подземный паркинг",
           "Ворота грузового двора")

# -------------------------------------------------------------------------- #
# Стили: тёмная консоль оператора
# -------------------------------------------------------------------------- #
CSS = """
<style>
/* ---- шапка ---- */
.topbar { display:flex; align-items:center; justify-content:space-between;
  gap:1rem; padding:.9rem 1.4rem; margin-bottom:1rem; border-radius:16px;
  background:linear-gradient(120deg,#0f1c33 0%,#13244a 100%);
  border:1px solid #223355; }
.brand { display:flex; align-items:center; gap:.8rem; }
.brand .logo { width:46px; height:46px; border-radius:12px;
  background:linear-gradient(135deg,#38bdf8,#2563eb); display:flex;
  align-items:center; justify-content:center; font-size:1.5rem; }
.brand .name { font-weight:800; font-size:1.25rem; letter-spacing:.02em; }
.brand .tag  { color:#7d8db5; font-size:.78rem; margin-top:.1rem; }
.sysinfo { text-align:right; color:#9fb0d6; font-size:.8rem; line-height:1.5; }
.sysinfo b { color:#4ade80; letter-spacing:.05em; }

/* ---- большие экраны решения ---- */
.banner { display:flex; align-items:center; justify-content:space-between;
  gap:1.4rem; padding:1.1rem 1.5rem; border-radius:20px; margin:.2rem 0 1rem;
  border:1px solid; flex-wrap:wrap; }
.banner.allow { background:linear-gradient(100deg,rgba(34,197,94,.20),rgba(34,197,94,.05));
  border-color:rgba(74,222,128,.55); }
.banner.deny  { background:linear-gradient(100deg,rgba(239,68,68,.20),rgba(239,68,68,.05));
  border-color:rgba(248,113,113,.55); }
.banner.manual{ background:linear-gradient(100deg,rgba(245,158,11,.20),rgba(245,158,11,.05));
  border-color:rgba(251,191,36,.55); }
.banner-left { display:flex; align-items:center; gap:1.2rem; }
.banner-icon { font-size:2.6rem; }
.banner-title { font-weight:800; font-size:1.45rem; letter-spacing:.06em; }
.banner.allow  .banner-title { color:#4ade80; }
.banner.deny   .banner-title { color:#f87171; }
.banner.manual .banner-title { color:#fbbf24; }
.banner-sub { color:#a7b6d9; font-size:.86rem; margin-top:.15rem; }

/* ---- номер как настоящая табличка ---- */
.plate-view { display:inline-flex; align-items:center; gap:.7rem;
  background:#f4f6f8; color:#0b1220; border-radius:10px;
  border:3px solid #0b1220; padding:.4rem 1rem; margin-top:.55rem;
  font-family:'Menlo','Consolas','DejaVu Sans Mono',monospace;
  font-weight:800; font-size:1.55rem; letter-spacing:.14em; }
.plate-view.small { font-size:1.05rem; padding:.25rem .7rem; }
.plate-region { display:flex; flex-direction:column; align-items:center;
  border-left:3px solid #0b1220; padding-left:.6rem; font-size:.95rem;
  letter-spacing:.05em; line-height:1.05; }
.plate-region .flag { font-size:.6rem; letter-spacing:0; color:#2563eb;
  border-top:2px solid #2563eb; padding-top:.1rem; margin-bottom:.1rem; }

/* ---- индикация шлагбаума ---- */
.gate { position:relative; min-width:150px; height:86px; }
.gate .post { position:absolute; left:10px; bottom:6px; width:12px; height:58px;
  background:#8b99b5; border-radius:4px;
  box-shadow:inset -3px 0 0 rgba(0,0,0,.25); }
.gate .arm { position:absolute; left:16px; bottom:54px; width:112px; height:10px;
  border-radius:5px; transform-origin:0 50%;
  background:repeating-linear-gradient(45deg,#ef4444 0 14px,#f1f5f9 14px 28px);
  box-shadow:0 3px 8px rgba(0,0,0,.45); }
.gate.open .arm { animation:arm-up 2.2s ease forwards; }
@keyframes arm-up { to { transform:rotate(-78deg); } }
.gate .lamp { position:absolute; left:4px; top:0; width:22px; height:22px;
  border-radius:50%; border:2px solid rgba(255,255,255,.15); }
.lamp.g { background:#22c55e; box-shadow:0 0 16px #22c55e; }
.lamp.r { background:#ef4444; box-shadow:0 0 16px #ef4444;
  animation:blink 1s infinite; }
.lamp.a { background:#f59e0b; box-shadow:0 0 16px #f59e0b;
  animation:blink 1.3s infinite; }
@keyframes blink { 50% { opacity:.3; } }
.gate-caption { text-align:center; color:#7d8db5; font-size:.7rem;
  margin-top:.3rem; }

/* ---- карточки и статистика ---- */
.card { background:#111a2e; border:1px solid #223355; border-radius:16px;
  padding:1rem 1.2rem; margin-bottom:.8rem; }
.card h4 { margin:0 0 .6rem; font-size:.95rem; color:#c7d3ee;
  letter-spacing:.05em; text-transform:uppercase; }
.stat-row { display:flex; gap:.9rem; flex-wrap:wrap; margin-bottom:1rem; }
.stat-card { flex:1 1 180px; background:#111a2e; border:1px solid #223355;
  border-radius:16px; padding:.9rem 1.1rem; }
.stat-card .v { font-size:1.8rem; font-weight:800; }
.stat-card .k { color:#7d8db5; font-size:.78rem; letter-spacing:.05em;
  text-transform:uppercase; margin-top:.1rem; }
.stat-card.allow .v  { color:#4ade80; }
.stat-card.deny .v   { color:#f87171; }
.stat-card.manual .v { color:#fbbf24; }

/* ---- шкала уверенности ---- */
.meter { height:12px; border-radius:8px; background:#1d2a47; overflow:hidden;
  margin:.35rem 0 .1rem; }
.meter > div { height:100%; border-radius:8px; }
.meter-row { display:flex; justify-content:space-between; color:#9fb0d6;
  font-size:.8rem; margin-bottom:.5rem; }

/* ---- чипы белого списка ---- */
.chip { display:inline-flex; align-items:center; gap:.55rem;
  background:#141c30; border:1px solid #2a3a5e; padding:.35rem .85rem;
  border-radius:999px; margin:.18rem .18rem 0 0;
  font-family:'Menlo','Consolas',monospace; font-weight:700; }
.chip .dot { width:8px; height:8px; border-radius:50%; background:#4ade80; }

/* ---- журнал ---- */
table.log { width:100%; border-collapse:collapse; font-size:.85rem; }
table.log th { text-align:left; color:#7d8db5; font-weight:600;
  padding:.5rem .6rem; border-bottom:1px solid #223355;
  font-size:.75rem; text-transform:uppercase; letter-spacing:.05em; }
table.log td { padding:.5rem .6rem; border-bottom:1px solid #18233c;
  color:#c7d3ee; }
table.log tr:hover td { background:#141c30; }
.badge { display:inline-block; padding:.15rem .6rem; border-radius:999px;
  font-size:.72rem; font-weight:800; letter-spacing:.04em; }
.badge.allow  { background:rgba(34,197,94,.15); color:#4ade80;
  border:1px solid rgba(74,222,128,.4); }
.badge.deny   { background:rgba(239,68,68,.15); color:#f87171;
  border:1px solid rgba(248,113,113,.4); }
.badge.manual { background:rgba(245,158,11,.15); color:#fbbf24;
  border:1px solid rgba(251,191,36,.4); }
.mono { font-family:'Menlo','Consolas',monospace; }

/* ---- мелочи ---- */
.hint { color:#7d8db5; font-size:.8rem; }
.footer { color:#5d6e96; font-size:.75rem; text-align:center;
  padding:1.2rem 0 .4rem; }
</style>
"""


# -------------------------------------------------------------------------- #
# Кэшируемые ресурсы
# -------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Загружаю детектор YOLO и CRNN…")
def get_pipeline() -> PlatePipeline:
    return PlatePipeline()


@st.cache_data(ttl=5)
def load_whitelist() -> list[str]:
    con = access_db.connect(DB_PATH)
    try:
        return access_db.get_whitelist(con)
    finally:
        con.close()


def whitelist_set() -> set[str]:
    return set(load_whitelist())


# -------------------------------------------------------------------------- #
# Мелкие помощники
# -------------------------------------------------------------------------- #
def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def draw_box(frame_bgr: np.ndarray, box, text: str = "") -> np.ndarray:
    out = frame_bgr.copy()
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 3)
    if text:
        cv2.rectangle(out, (x1, max(0, y1 - 32)), (x1 + 14 * len(text), y1),
                      (0, 220, 0), -1)
        cv2.putText(out, text, (x1 + 4, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 2)
    return out


def is_video(name: str) -> bool:
    return Path(name).suffix.lower().lstrip(".") in VIDEO_EXTS


def pretty_plate(text: str) -> str:
    """'O777EE34' -> 'O 777 EE 34' для читаемого показа."""
    t = text.strip()
    if len(t) in (8, 9) and t[:1].isalpha() and t[1:4].isdigit() \
            and t[4:6].isalpha() and t[6:].isdigit():
        return f"{t[0]} {t[1:4]} {t[4:6]} {t[6:]}"
    return t or "—"


def plate_html(text: str, small: bool = False) -> str:
    """HTML «настоящей» таблички номера."""
    cls = "plate-view small" if small else "plate-view"
    t = text.strip()
    region = t[6:] if len(t) in (8, 9) else ""
    region_html = (f'<span class="plate-region"><span class="flag">RUS</span>'
                   f"{region or '—'}</span>") if region else ""
    return (f'<span class="{cls}">{pretty_plate(t)}{region_html}</span>')


def gate_html(decision: str) -> str:
    """Индикация шлагбаума. Сайт шлагбаумом НЕ управляет — это только экран."""
    if decision == DECISION_ALLOW:
        lamp, arm, label = "g", "open", "ШЛАГБАУМ: ОТКРЫТ"
    elif decision == DECISION_DENY:
        lamp, arm, label = "r", "", "ШЛАГБАУМ: ЗАКРЫТ"
    else:
        lamp, arm, label = "a", "", "ОЖИДАНИЕ ОПЕРАТОРА"
    return f"""
<div><div class="gate {arm}">
  <div class="lamp {lamp}"></div><div class="post"></div><div class="arm"></div>
</div><div class="gate-caption">{label}</div></div>"""


def meter_html(value: float, threshold: float | None = None,
               label: str = "уверенность OCR") -> str:
    pct = max(0.0, min(1.0, value)) * 100
    if threshold is None or value >= threshold:
        color = "#38bdf8"
    else:
        color = "#f59e0b"
    thr = f"порог {threshold:.2f}" if threshold is not None else ""
    return (f'<div class="meter-row"><span>{label}</span>'
            f"<span><b>{value:.2f}</b> {thr}</span></div>"
            f'<div class="meter"><div style="width:{pct:.0f}%;'
            f'background:linear-gradient(90deg,{color}88,{color});"></div></div>')


def banner_html(decision: str, plate: str, reason: str,
                source: str = "") -> str:
    """Большой экран решения, как на посту въезда."""
    icon = {DECISION_ALLOW: "✅", DECISION_DENY: "⛔",
            DECISION_MANUAL: "⚠️"}[decision]
    title = {"ALLOW": "ДОСТУП РАЗРЕШЁН", "DENY": "ДОСТУП ЗАПРЕЩЁН",
             "MANUAL": "ТРЕБУЕТСЯ ПРОВЕРКА ОПЕРАТОРА"}[decision]
    sub = reason + (f" · {source}" if source else "")
    return f"""
<div class="banner {decision.lower()}">
  <div class="banner-left">
    <div class="banner-icon">{icon}</div>
    <div>
      <div class="banner-title">{title}</div>
      {plate_html(plate)}
      <div class="banner-sub">{sub}</div>
    </div>
  </div>
  {gate_html(decision)}
</div>"""


# -------------------------------------------------------------------------- #
# Шапка
# -------------------------------------------------------------------------- #
def render_header() -> None:
    now = datetime.now()
    st.markdown(f"""
<div class="topbar">
  <div class="brand">
    <div class="logo">🅿️</div>
    <div>
      <div class="name">SmartParking <span style="color:#38bdf8">AI</span></div>
      <div class="tag">консоль оператора · контроль въезда по номеру</div>
    </div>
  </div>
  <div class="sysinfo">
    <div><b>● СИСТЕМА АКТИВНА</b></div>
    <div>{now:%d.%m.%Y} · сервер решений на месте</div>
    <div>шлагбаумом сайт не управляет — показ решения на экране</div>
  </div>
</div>""", unsafe_allow_html=True)


# -------------------------------------------------------------------------- #
# Вкладка 1: контроль въезда
# -------------------------------------------------------------------------- #
def process_upload(data: bytes, name: str, conf: float, margin: float,
                   n_frames: int) -> dict | None:
    """Обрабатывает загруженный файл. Кэширует результат в session_state,
    чтобы смена косметических настроек не пересчитывала видео заново."""
    key = (name, len(data), conf, margin, n_frames,
           tuple(sorted(whitelist_set())))
    if st.session_state.get("result_key") == key:
        return st.session_state["result"]

    pipe = get_pipeline()
    if is_video(name):
        with tempfile.NamedTemporaryFile(suffix=Path(name).suffix,
                                        delete=False) as tmp:
            tmp.write(data)
            video_path = tmp.name
        try:
            bar = st.progress(0.0, text="Читаю кадры с камеры…")
            r = pipe.process_video(video_path, whitelist_set(), conf,
                                   margin, max_frames=n_frames,
                                   progress=lambda p: bar.progress(
                                       p, text="Читаю кадры с камеры…"))
            bar.empty()
            res = {"kind": "video", "r": r, "frame": None}
        finally:
            Path(video_path).unlink(missing_ok=True)
    else:
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            st.error("Не удалось открыть файл как изображение.")
            return None
        with st.spinner("Ищу номер и читаю…"):
            r = pipe.process_frame(frame, whitelist_set(), conf, margin)
        res = {"kind": "photo", "r": r, "frame": frame}

    st.session_state["result_key"] = key
    st.session_state["result"] = res
    return res


def render_result(res: dict, conf: float, camera: str) -> None:
    kind, r, frame = res["kind"], res["r"], res["frame"]
    st.markdown(banner_html(r.decision, r.text, r.reason, camera),
                unsafe_allow_html=True)

    left, right = st.columns([5, 3])
    with left:
        st.markdown('<div class="card"><h4>Кадр камеры · '
                    f'{camera}</h4>', unsafe_allow_html=True)
        if kind == "video" and r.best is not None and r.best_frame is not None:
            shown = draw_box(r.best_frame, r.best.box, r.best.text)
            st.image(bgr_to_pil(shown),
                     caption=f"Лучший кадр · детектор {r.best.det_conf:.2f} · "
                             f"решение по {r.frames_read} чтениям, "
                             "а не по одному кадру")
        elif kind == "photo" and r.box is not None:
            shown = draw_box(frame, r.box, r.text)
            st.image(bgr_to_pil(shown),
                     caption=f"Детектор: YOLO11n, вход 640, "
                             f"уверенность рамки {r.det_conf:.2f}")
        elif kind == "photo":
            st.image(bgr_to_pil(frame),
                     caption="Номер детектором не найден — это ручная "
                             "проверка, а не отказ.")
        else:
            st.info("Ни в одном прочитанном кадре не удалось прочитать "
                    "номер — это ручная проверка, а не отказ.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card"><h4>Распознавание</h4>',
                    unsafe_allow_html=True)
        st.markdown(meter_html(r.ocr_conf, conf), unsafe_allow_html=True)
        if kind == "video":
            share = r.votes / max(r.frames_read, 1)
            st.markdown(meter_html(share, None,
                                   label="кадров за этот текст"),
                        unsafe_allow_html=True)
        st.markdown(f"""
<div class="hint">
  {'чтений: ' + str(r.frames_read) + ' · голосовало: ' + str(r.votes)
   if kind == 'video' else 'одно чтение кадра'}
</div>""", unsafe_allow_html=True)
        if kind == "photo" and r.crop is not None:
            st.image(bgr_to_pil(r.crop),
                     caption="кроп, который читала CRNN")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
<div class="stat-row">
  <div class="stat-card"><div class="v">{r.elapsed_ms:.0f} мс</div>
    <div class="k">от кадра до решения</div></div>
  <div class="stat-card"><div class="v">{len(load_whitelist())}</div>
    <div class="k">резидентов в списке</div></div>
  <div class="stat-card"><div class="v">{conf:.2f}</div>
    <div class="k">порог уверенности</div></div>
</div>""", unsafe_allow_html=True)


def tab_control(conf: float, margin: float, n_frames: int) -> None:
    camera = st.selectbox("Пост въезда", CAMERAS, index=0)
    st.caption(f"Источник кадра: {camera}. Загрузите снимок или короткое "
               "видео с этого поста.")
    uploaded = st.file_uploader(
        "Кадр с камеры (фото или видео)",
        type=list(PHOTO_EXTS) + list(VIDEO_EXTS),
        label_visibility="collapsed",
        help="Фото: PNG/JPG. Видео: MP4/AVI/MOV — решение будет принято по "
             "нескольким кадрам, а не по первому.")

    if uploaded is None:
        st.markdown('<div class="card"><h4>Ожидание кадра</h4>'
                    '<div class="hint">Пост в режиме ожидания. Загрузите '
                    'фотографию или короткое видео — система найдёт номер, '
                    'прочитает его и покажет решение на въезде.</div></div>',
                    unsafe_allow_html=True)
        return

    data = uploaded.read()
    res = process_upload(data, uploaded.name, conf, margin, n_frames)
    if res is not None:
        render_result(res, conf, camera)
        kind, r = res["kind"], res["r"]
        con = access_db.connect(DB_PATH)
        try:
            access_db.add_history(
                con, source="видео" if kind == "video" else "фото",
                plate=r.text, ocr_conf=r.ocr_conf,
                det_conf=(r.best.det_conf if kind == "video" and r.best
                          else r.det_conf),
                votes=(r.votes if kind == "video" else 1),
                decision=r.decision, reason=r.reason, time_ms=r.elapsed_ms)
        finally:
            con.close()


# -------------------------------------------------------------------------- #
# Вкладка 2: резиденты
# -------------------------------------------------------------------------- #
def tab_residents() -> None:
    st.markdown('<div class="card"><h4>Добавить резидента</h4>',
                unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    plate_in = col1.text_input("Номер", placeholder="A123BC777 или А123ВС777",
                               label_visibility="collapsed")
    norm_in = normalize_plate(plate_in)

    # живая проверка ввода: сразу видно, почему номер не примут
    if plate_in.strip() and not PLATE_RE.fullmatch(norm_in):
        st.markdown(
            '<div class="hint" style="color:#fbbf24;">'
            f"«{plate_in.strip()}» → «{norm_in}»: не похож на российский номер. "
            "В номерах используются только буквы АВЕКМНОРСТУХ "
            "(латиницей ABEKMHOCTYX); формат: буква · 3 цифры · 2 буквы · "
            "регион из 2–3 цифр. Буквы И и I в номерах не бывает.</div>",
            unsafe_allow_html=True)
    elif plate_in.strip():
        st.markdown(f'<div class="hint">будет добавлен как '
                    f"<b>{norm_in}</b></div>", unsafe_allow_html=True)

    if col2.button("Добавить", use_container_width=True):
        if not norm_in:
            st.warning("Введите номер.")
        else:
            con = access_db.connect(DB_PATH)
            try:
                ok, msg = access_db.add_plate(con, plate_in)
            finally:
                con.close()
            if ok:
                load_whitelist.clear()
                st.toast(msg, icon="✅")
                st.rerun()
            st.error(msg)   # причина отказа остаётся на экране, а не в тосте

    # быстрое добавление последнего прочитанного номера (для сценария А)
    last = st.session_state.get("result")
    if last is not None and last["r"].text \
            and PLATE_RE.fullmatch(last["r"].text):
        if st.button(f"➕ Добавить последний прочитанный: {last['r'].text}",
                     use_container_width=True,
                     help="Номер, который система прочитала на вкладке "
                          "«Контроль въезда». Для сценария А вносите именно "
                          "то, что реально прочитано, а не вручную набитую "
                          "строку."):
            con = access_db.connect(DB_PATH)
            try:
                ok, msg = access_db.add_plate(con, last["r"].text)
            finally:
                con.close()
            if ok:
                load_whitelist.clear()
                st.toast(msg, icon="✅")
                st.rerun()
            st.error(msg)

    plates = load_whitelist()
    if plates:
        chips = "".join(f'<span class="chip"><span class="dot"></span>'
                        f"{p}</span>" for p in plates)
        st.markdown(chips, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><h4>Удалить резидента</h4>',
                    unsafe_allow_html=True)
        del_col1, del_col2 = st.columns([3, 1])
        to_delete = del_col1.selectbox(
            "Номер", plates, label_visibility="collapsed")
        if del_col2.button("Удалить", use_container_width=True):
            con = access_db.connect(DB_PATH)
            try:
                _, msg = access_db.remove_plate(con, to_delete)
            finally:
                con.close()
            load_whitelist.clear()
            st.toast(msg, icon="🗑️")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="hint">Список пуст: каждый уверенно '
                    'прочитанный номер получит отказ.</div></div>',
                    unsafe_allow_html=True)


# -------------------------------------------------------------------------- #
# Вкладка 3: журнал
# -------------------------------------------------------------------------- #
def tab_log() -> None:
    con = access_db.connect(DB_PATH)
    try:
        stats = access_db.get_stats(con, datetime.now().strftime("%Y-%m-%d"))
        rows = access_db.get_history(con, limit=30)
    finally:
        con.close()

    allow = stats.get(DECISION_ALLOW, {}).get("count", 0)
    deny = stats.get(DECISION_DENY, {}).get("count", 0)
    manual = stats.get(DECISION_MANUAL, {}).get("count", 0)
    total_ms = [s["avg_ms"] for s in stats.values() if s["avg_ms"]]
    avg_ms = sum(total_ms) / len(total_ms) if total_ms else 0.0

    st.markdown(f"""
<div class="stat-row">
  <div class="stat-card allow"><div class="v">{allow}</div>
    <div class="k">пущено сегодня</div></div>
  <div class="stat-card deny"><div class="v">{deny}</div>
    <div class="k">отказано сегодня</div></div>
  <div class="stat-card manual"><div class="v">{manual}</div>
    <div class="k">на ручной проверке</div></div>
  <div class="stat-card"><div class="v">{avg_ms:.1f} мс</div>
    <div class="k">среднее время решения</div></div>
</div>""", unsafe_allow_html=True)

    if not rows:
        st.markdown('<div class="card"><h4>Журнал пуст</h4>'
                    '<div class="hint">Обработайте кадр на вкладке '
                    '«Контроль въезда» — записи появятся здесь.</div></div>',
                    unsafe_allow_html=True)
        return

    badge = {DECISION_ALLOW: "allow", DECISION_DENY: "deny",
             DECISION_MANUAL: "manual"}
    trs = []
    for r in rows:
        conf = f"{r['ocr_conf']:.2f}" if r["ocr_conf"] is not None else "—"
        trs.append(
            f"<tr><td>{r['ts'][11:19]}</td><td>{r['source']}</td>"
            f"<td class='mono'><b>{r['plate'] or '—'}</b></td>"
            f"<td>{conf}</td><td>{r['votes'] or 1}</td>"
            f"<td><span class='badge {badge[r['decision']]}'>"
            f"{DECISION_LABELS[r['decision']]}</span></td>"
            f"<td class='hint'>{r['reason']}</td>"
            f"<td>{round(r['time_ms'])} мс</td></tr>")
    st.markdown(
        '<div class="card"><h4>Последние решения</h4><table class="log">'
        "<tr><th>время</th><th>вход</th><th>номер</th><th>ocr</th>"
        "<th>кадров</th><th>решение</th><th>причина</th><th>время решения</th>"
        f"</tr>{''.join(trs)}</table></div>", unsafe_allow_html=True)

    if st.button("Очистить журнал", type="primary"):
        con = access_db.connect(DB_PATH)
        try:
            con.execute("DELETE FROM history")
            con.commit()
        finally:
            con.close()
        st.rerun()


# -------------------------------------------------------------------------- #
# Вкладка 4: настройки
# -------------------------------------------------------------------------- #
def tab_settings() -> None:
    st.markdown('<div class="card"><h4>Пороги решения</h4>'
                '<div class="hint">Одно слабое чтение не должно давать '
                '«пустить». Подберите порог на своих кадрах и верните '
                'значение по умолчанию 0.85 перед показом.</div></div>',
                unsafe_allow_html=True)
    conf = st.slider("Порог уверенности OCR", 0.50, 0.95,
                     DEFAULT_CONF_THRESHOLD, 0.01,
                     help="Чтения ниже порога уходят на ручную проверку.")
    margin = st.slider("Запас вокруг рамки", 0.03, 0.12,
                       DEFAULT_MARGIN, 0.01, format="%.2f",
                       help="Доля стороны рамки с каждой стороны. Если текст "
                            "на кропах разваливается, правьте сначала это, "
                            "а не веса.")
    n_frames = st.slider("Кадров из видео", 3, 20, VIDEO_MAX_FRAMES,
                         help="Решение по видео принимается по нескольким "
                              "кадрам: побеждает текст с наибольшей суммой "
                              "уверенностей.")
    st.markdown(f"""
<div class="card"><h4>Модели в системе</h4>
<table class="log">
<tr><td>детектор номера</td><td>YOLO11n, вход 640, mAP@50 0.984</td></tr>
<tr><td>обучение детектора</td><td><span class="badge manual">3 эпохи,
не финальное сравнение</span></td></tr>
<tr><td>чтение номера</td><td>CRNN, Exact Match 98,9% на val</td></tr>
<tr><td>распознаватель</td><td>crnn_best.pt (PaddleOCR не используется)</td></tr>
<tr><td>база доступа</td><td class="mono">data/processed/access.sqlite</td></tr>
</table></div>""", unsafe_allow_html=True)
    return conf, margin, n_frames


# -------------------------------------------------------------------------- #
# Страница
# -------------------------------------------------------------------------- #
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    render_header()

    tab_control_, tab_res_, tab_log_, tab_set_ = st.tabs(
        ["🛂 Контроль въезда", "👥 Резиденты", "🗂 Журнал", "⚙️ Настройки"])

    with tab_set_:
        conf, margin, n_frames = tab_settings()

    with tab_control_:
        tab_control(conf, margin, n_frames)

    with tab_res_:
        tab_residents()

    with tab_log_:
        tab_log()

    st.markdown(
        '<div class="footer">SmartParking AI · учебное демо · детектор '
        "YOLO11n (3 эпохи, не финальное сравнение), чтение CRNN (Exact Match "
        "98,9% на val) · шлагбаумом сайт не управляет</div>",
        unsafe_allow_html=True)


main()

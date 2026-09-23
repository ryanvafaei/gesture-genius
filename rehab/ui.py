"""
Act: readable screens and pictures.

Text     Pillow draws all text in Atkinson Hyperlegible (assets/fonts) at a
         large size. Text is collected during a frame and drawn in one go, so
         the image is converted to Pillow and back only once per frame.
         Without Pillow or the font it falls back to OpenCV text.
Screens  card     full screen: greetings, questions, exercise introductions
         summary  one highlight, the garden, the closing message
         garden   the garden with today's growth (short animation)
         The exercise screen itself (camera + panel) is drawn by Act.Display,
         which adds a small activity icon, the watering can and the coach's
         face from here.
Pictures assets/icons/<icon>.png, assets/garden/<plant>_<stage>.png,
         assets/garden/{bee,butterfly,can,bed}.png are used when present (PNG
         with transparency, e.g. exported from Figma); otherwise simple drawn
         shapes in calm, natural colours - realistic proportions, no cartoon
         faces on plants, nothing childish.

Nothing here decides anything: it draws what the view dict says.
"""

import math
from functools import lru_cache

import cv2
import numpy as np

from rehab import config

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:            # pragma: no cover - Pillow is in requirements.txt
    Image = ImageDraw = ImageFont = None

# colours (BGR)
BG = (236, 241, 245)
INK = (40, 40, 40)
SOFT = (110, 110, 110)
WHITE = (255, 255, 255)
GREEN = (96, 150, 76)
GREY = (160, 160, 160)
SKY = (238, 228, 205)
SOIL = (60, 86, 120)
SOIL_DARK = (45, 66, 95)
GRASS = (110, 180, 140)
STEM = (70, 130, 80)
LEAF = (80, 160, 100)
WATER = (210, 150, 90)
CAN = (150, 140, 120)
PLANT_COLORS = {
    "rose": ((80, 60, 200), (60, 40, 150)),
    "tulip": ((60, 140, 235), (40, 100, 190)),
    "sunflower": ((50, 200, 240), (40, 70, 110)),
    "lavender": ((200, 120, 150), (160, 90, 120)),
}
MOODS = ("neutral", "happy", "encouraging")

FONT_PX_PER_SCALE = 34          # cv2 scale 1.0 is about 34 px of text


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _font(size, bold):
    if ImageFont is None:
        return None
    path = config.FONT_BOLD if bold else config.FONT_REGULAR
    for candidate in (path, config.FONT_REGULAR, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def pillow_available():
    return _font(20, False) is not None


class TextLayer:
    """
    Collects text for one frame, then draws it all at once with apply().
    Positions are the left end of the baseline, like cv2.putText.
    """

    def __init__(self):
        self.ops = []

    def measure(self, text, size, bold=False):
        font = _font(size, bold)
        if font is None:
            scale = size / FONT_PX_PER_SCALE
            return cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2 if bold else 1)[0][0]
        return font.getlength(text)

    def wrap(self, text, size, bold, max_width):
        words = text.split()
        lines, line = [], ""
        for w in words:
            trial = f"{line} {w}".strip()
            if line and self.measure(trial, size, bold) > max_width:
                lines.append(line)
                line = w
            else:
                line = trial
        if line:
            lines.append(line)
        return lines

    def add(self, text, xy, size, color=INK, bold=False, max_width=None, line_gap=1.3,
            align="left", max_lines=None):
        """Queue text (wrapped to max_width). Returns the y below the last line."""
        if not text:
            return xy[1]
        lines = self.wrap(text, size, bold, max_width) if max_width else [text]
        if max_lines:
            lines = lines[:max_lines]
        x, y = xy
        for line in lines:
            lx = x
            if align == "center":
                lx = x - self.measure(line, size, bold) / 2
            elif align == "right":
                lx = x - self.measure(line, size, bold)
            self.ops.append((line, (int(lx), int(y)), size, color, bold))
            y += size * line_gap
        return y - size * line_gap + size * 0.45

    def drop(self, x0, x1):
        """Forget text starting between x0 and x1 (e.g. covered by the pause screen)."""
        self.ops = [op for op in self.ops if not x0 <= op[1][0] < x1]

    def apply(self, img):
        if not self.ops:
            return img
        if _font(20, False) is None:
            for text, (x, y), size, color, bold in self.ops:
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, size / FONT_PX_PER_SCALE,
                            color, 2 if bold else 1, cv2.LINE_AA)
            self.ops = []
            return img
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for text, (x, y), size, color, bold in self.ops:
            draw.text((x, y), text, font=_font(size, bold), fill=tuple(reversed(color)),
                      anchor="ls")
        self.ops = []
        img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        return img


# ---------------------------------------------------------------------------
# PNG assets (optional)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def _png(relative):
    path = config.ASSETS_DIR / relative
    if not path.is_file():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3:
        return None
    if img.shape[2] == 3:
        img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    return img


def paste(img, png, center, size, alpha=1.0):
    """Paste an RGBA picture with its longest side = size, centred at center."""
    h, w = png.shape[:2]
    k = size / max(h, w)
    png = cv2.resize(png, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
    h, w = png.shape[:2]
    x0, y0 = int(center[0] - w / 2), int(center[1] - h / 2)
    X0, Y0 = max(0, x0), max(0, y0)
    X1, Y1 = min(img.shape[1], x0 + w), min(img.shape[0], y0 + h)
    if X1 <= X0 or Y1 <= Y0:
        return
    part = png[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    a = part[:, :, 3:4].astype(np.float32) / 255.0 * alpha
    region = img[Y0:Y1, X0:X1].astype(np.float32)
    img[Y0:Y1, X0:X1] = (region * (1 - a) + part[:, :, :3].astype(np.float32) * a).astype(np.uint8)


def _aa(img, pts, color):
    cv2.fillPoly(img, [np.asarray(pts, np.int32)], color, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Coach face
# ---------------------------------------------------------------------------

def draw_face(img, center, r, mood="neutral", character=None):
    """A calm, simple face with three expressions: neutral, happy, encouraging."""
    av = (character or {}).get("avatar", {})
    skin = tuple(reversed(av.get("skin", (224, 196, 172))))
    hair = tuple(reversed(av.get("hair", (120, 130, 140))))
    back = tuple(reversed(av.get("background", (206, 226, 214))))
    cx, cy = int(center[0]), int(center[1])
    cv2.circle(img, (cx, cy), int(r * 1.12), back, -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy - int(r * 0.1)), (int(r * 0.98), int(r * 0.95)), 0, 180, 360,
                hair, -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy + int(r * 0.08)), (int(r * 0.82), int(r * 0.9)), 0, 0, 360,
                skin, -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy - int(r * 0.55)), (int(r * 0.85), int(r * 0.42)), 0, 180, 360,
                hair, -1, cv2.LINE_AA)
    ink = (60, 60, 70)
    ex, ey, er = int(r * 0.32), cy - int(r * 0.05), max(2, int(r * 0.08))
    for sx in (-1, 1):
        x = cx + sx * ex
        if mood == "happy":
            cv2.ellipse(img, (x, ey + er), (er * 2, er * 2), 0, 200, 340, ink,
                        max(2, er // 2), cv2.LINE_AA)
        else:
            cv2.circle(img, (x, ey), er, ink, -1, cv2.LINE_AA)
        if mood == "encouraging":
            cv2.line(img, (x - er * 2, ey - er * 3 - sx * 2), (x + er * 2, ey - er * 3 + sx * 2),
                     ink, max(2, er // 2), cv2.LINE_AA)
    my = cy + int(r * 0.38)
    if mood == "happy":
        cv2.ellipse(img, (cx, my - int(r * 0.08)), (int(r * 0.32), int(r * 0.22)), 0, 10, 170,
                    ink, max(2, int(r * 0.06)), cv2.LINE_AA)
    elif mood == "encouraging":
        cv2.ellipse(img, (cx, my - int(r * 0.05)), (int(r * 0.26), int(r * 0.14)), 0, 15, 165,
                    ink, max(2, int(r * 0.05)), cv2.LINE_AA)
    else:
        cv2.ellipse(img, (cx, my - int(r * 0.05)), (int(r * 0.22), int(r * 0.08)), 0, 20, 160,
                    ink, max(2, int(r * 0.05)), cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Activity icons
# ---------------------------------------------------------------------------

def draw_icon(img, name, center, size, color=INK):
    """One-colour pictogram (or assets/icons/<name>.png)."""
    png = _png(f"icons/{name}.png")
    if png is not None:
        paste(img, png, center, size)
        return
    cx, cy = int(center[0]), int(center[1])
    s = size / 100.0
    t = max(2, int(5 * s))

    def P(x, y):
        return int(cx + x * s), int(cy + y * s)

    if name == "cup":
        _aa(img, [P(-32, -22), P(22, -22), P(16, 34), P(-26, 34)], color)
        cv2.ellipse(img, P(26, 4), (int(14 * s), int(16 * s)), 0, -90, 90, color, t, cv2.LINE_AA)
        for x in (-16, -2, 12):
            cv2.ellipse(img, P(x, -36), (int(4 * s), int(9 * s)), 0, 90, 270, color, t, cv2.LINE_AA)
    elif name == "trowel":
        _aa(img, [P(-6, -46), P(20, -12), P(8, 10), P(-18, 10), P(-26, -12)], color)
        cv2.line(img, P(-5, 10), P(-5, 22), color, t, cv2.LINE_AA)
        _aa(img, [P(-13, 22), P(3, 22), P(3, 48), P(-13, 48)], color)
    elif name == "pot":
        _aa(img, [P(-34, -8), P(34, -8), P(30, 34), P(-30, 34)], color)
        cv2.line(img, P(-40, -16), P(40, -16), color, t, cv2.LINE_AA)
        cv2.circle(img, P(0, -24), int(5 * s), color, -1, cv2.LINE_AA)
        for sx in (-1, 1):
            cv2.line(img, P(sx * 34, 2), P(sx * 46, 2), color, t, cv2.LINE_AA)
    elif name == "book":
        _aa(img, [P(-44, -28), P(-3, -20), P(-3, 32), P(-44, 24)], color)
        _aa(img, [P(44, -28), P(3, -20), P(3, 32), P(44, 24)], color)
    elif name == "button":
        cv2.circle(img, P(0, 0), int(38 * s), color, -1, cv2.LINE_AA)
        bg = tuple(int(c) for c in img[min(img.shape[0] - 1, cy + int(55 * s)),
                                         min(img.shape[1] - 1, cx)])
        for x, y in ((-10, -10), (10, -10), (-10, 10), (10, 10)):
            cv2.circle(img, P(x, y), max(2, int(6 * s)), bg, -1, cv2.LINE_AA)
    elif name == "phone":
        cv2.rectangle(img, P(-24, -44), P(24, 44), color, -1, cv2.LINE_AA)
        bg = tuple(int(c) for c in img[min(img.shape[0] - 1, cy + int(60 * s)),
                                         min(img.shape[1] - 1, cx)])
        cv2.rectangle(img, P(-18, -34), P(18, 26), bg, -1, cv2.LINE_AA)
        cv2.circle(img, P(0, 35), max(2, int(4 * s)), bg, -1, cv2.LINE_AA)
    else:
        cv2.circle(img, P(0, 0), int(36 * s), color, t, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Watering can (session progress)
# ---------------------------------------------------------------------------

def _rot(points, center, angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    cx, cy = center
    return [(cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c) for x, y in points]


def draw_can(img, center, size, sections=3, filled=0, tilt=0.0):
    """Watering can; its body has one section per exercise of today, filled ones blue."""
    png = _png("garden/can.png")
    if png is not None and sections <= 0:
        paste(img, png, center, size)
        return
    cx, cy = center
    s = size / 100.0
    body = [(cx - 30 * s, cy - 25 * s), (cx + 26 * s, cy - 25 * s),
            (cx + 26 * s, cy + 30 * s), (cx - 30 * s, cy + 30 * s)]
    spout = [(cx + 24 * s, cy + 10 * s), (cx + 58 * s, cy - 30 * s),
             (cx + 62 * s, cy - 25 * s), (cx + 26 * s, cy + 22 * s)]
    rose = [(cx + 55 * s, cy - 36 * s), (cx + 68 * s, cy - 24 * s)]
    t = max(2, int(5 * s))
    _aa(img, _rot(spout, (cx, cy), tilt), CAN)
    a, b = _rot(rose, (cx, cy), tilt)
    cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), CAN, int(8 * s), cv2.LINE_AA)
    _aa(img, _rot(body, (cx, cy), tilt), CAN)
    # sections, bottom up
    n = max(1, sections)
    inner_h = 49 * s
    for i in range(n):
        y1 = cy + 27 * s - inner_h * i / n
        y0 = cy + 27 * s - inner_h * (i + 1) / n + 2
        box = [(cx - 27 * s, y0), (cx + 23 * s, y0), (cx + 23 * s, y1), (cx - 27 * s, y1)]
        _aa(img, _rot(box, (cx, cy), tilt), WATER if i < filled else (225, 225, 225))
    handle = _rot([(cx - 30 * s, cy - 14 * s)], (cx, cy), tilt)[0]
    cv2.ellipse(img, (int(handle[0]), int(handle[1] + 10 * s)), (int(14 * s), int(20 * s)),
                tilt, 90, 270, CAN, t, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Garden
# ---------------------------------------------------------------------------

def draw_plant(img, base, height, plant, stage, scale=1.0):
    """One plant at a growth stage (0 seed .. 4 flower), base = (x, soil y)."""
    png = _png(f"garden/{plant}_{stage}.png")
    x, y = int(base[0]), int(base[1])
    h = height * scale
    if png is not None:
        paste(img, png, (x, int(y - h / 2)), h)
        return
    petal, center = PLANT_COLORS.get(plant, PLANT_COLORS["rose"])
    t = max(2, int(h * 0.025))
    if stage == 0:
        cv2.ellipse(img, (x, y - 2), (int(h * 0.16), int(h * 0.05)), 0, 180, 360, SOIL_DARK, -1,
                    cv2.LINE_AA)
        cv2.ellipse(img, (x, y - int(h * 0.05)), (int(h * 0.035), int(h * 0.025)), -20, 0, 360,
                    (70, 110, 150), -1, cv2.LINE_AA)
        return
    stem_h = h * (0.25, 0.5, 0.75, 0.82)[min(stage, 4) - 1]
    if plant == "sunflower" and stage >= 3:
        stem_h *= 1.1
    top = (x, int(y - stem_h))
    cv2.line(img, (x, y), top, STEM, t + 1, cv2.LINE_AA)

    def leaf(frac, side, size):
        ly = int(y - stem_h * frac)
        cv2.ellipse(img, (x + side * int(h * size * 0.9), ly), (int(h * size), int(h * size * 0.38)),
                    -side * 30, 0, 360, LEAF, -1, cv2.LINE_AA)

    leaf(0.55 if stage == 1 else 0.3, -1, 0.07 if stage == 1 else 0.1)
    leaf(0.75 if stage == 1 else 0.45, 1, 0.07 if stage == 1 else 0.1)
    if stage >= 2:
        leaf(0.6, -1, 0.09)
        leaf(0.72, 1, 0.08)
    if stage == 3:
        if plant == "lavender":
            for i in range(5):
                cv2.circle(img, (x, top[1] - int(i * h * 0.03)), int(h * 0.018), center, -1,
                           cv2.LINE_AA)
        else:
            cv2.ellipse(img, (top[0], top[1] - int(h * 0.04)), (int(h * 0.04), int(h * 0.07)), 0,
                        0, 360, center if plant != "sunflower" else STEM, -1, cv2.LINE_AA)
            cv2.ellipse(img, (top[0], top[1] - int(h * 0.06)), (int(h * 0.03), int(h * 0.05)), 0,
                        0, 360, petal, -1, cv2.LINE_AA)
    elif stage >= 4:
        fx, fy = top
        if plant == "rose":
            for rr, col in ((0.1, petal), (0.07, center), (0.045, petal), (0.02, center)):
                cv2.circle(img, (fx, fy - int(h * 0.04)), int(h * rr), col, -1, cv2.LINE_AA)
        elif plant == "tulip":
            for dx, ang in ((-0.05, 15), (0.05, -15), (0.0, 0)):
                cv2.ellipse(img, (fx + int(h * dx), fy - int(h * 0.08)),
                            (int(h * 0.05), int(h * 0.1)), ang, 0, 360,
                            petal if dx else center, -1, cv2.LINE_AA)
        elif plant == "sunflower":
            for k in range(12):
                a = k * math.pi / 6
                px = fx + int(math.cos(a) * h * 0.1)
                py = fy - int(h * 0.05) + int(math.sin(a) * h * 0.1)
                cv2.ellipse(img, (px, py), (int(h * 0.055), int(h * 0.025)), math.degrees(a),
                            0, 360, petal, -1, cv2.LINE_AA)
            cv2.circle(img, (fx, fy - int(h * 0.05)), int(h * 0.065), center, -1, cv2.LINE_AA)
        else:   # lavender: a spike of small flowers
            for i in range(9):
                side = -1 if i % 2 else 1
                cv2.ellipse(img, (fx + side * int(h * 0.015), fy - int(i * h * 0.03)),
                            (int(h * 0.022), int(h * 0.016)), 0, 0, 360,
                            petal if i % 3 else center, -1, cv2.LINE_AA)


def draw_bee(img, center, size):
    png = _png("garden/bee.png")
    if png is not None:
        paste(img, png, center, size)
        return
    cx, cy = int(center[0]), int(center[1])
    s = size / 40.0
    for sx in (-1, 1):
        cv2.ellipse(img, (cx + int(sx * 5 * s), cy - int(9 * s)), (int(7 * s), int(10 * s)),
                    sx * 25, 0, 360, (250, 245, 235), -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (int(14 * s), int(9 * s)), 0, 0, 360, (40, 190, 235), -1, cv2.LINE_AA)
    for dx in (-5, 3):
        cv2.line(img, (cx + int(dx * s), cy - int(8 * s)), (cx + int(dx * s), cy + int(8 * s)),
                 (40, 40, 40), max(2, int(3 * s)), cv2.LINE_AA)


def draw_butterfly(img, center, size):
    png = _png("garden/butterfly.png")
    if png is not None:
        paste(img, png, center, size)
        return
    cx, cy = int(center[0]), int(center[1])
    s = size / 40.0
    for sx in (-1, 1):
        cv2.ellipse(img, (cx + int(sx * 10 * s), cy - int(6 * s)), (int(10 * s), int(12 * s)),
                    sx * 20, 0, 360, (190, 130, 90), -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx + int(sx * 8 * s), cy + int(9 * s)), (int(7 * s), int(8 * s)),
                    -sx * 20, 0, 360, (210, 160, 120), -1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - int(12 * s)), (cx, cy + int(14 * s)), (50, 50, 50), max(2, int(3 * s)),
             cv2.LINE_AA)


def _extra_positions(n, area, seed):
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = area
    return [(int(rng.uniform(x0, x1)), int(rng.uniform(y0, y1))) for _ in range(n)]


def draw_garden(img, rect, garden, grow=None, max_extras=8):
    """
    The garden in rect (x0, y0, x1, y1). grow = {"plot", "prev_stage", "progress",
    "new_bees", "new_butterflies"} animates today's growth: the plant scales and
    fades into its new stage, the watering can tips above it, new visitors fade in.
    """
    x0, y0, x1, y1 = [int(v) for v in rect]
    w, h = x1 - x0, y1 - y0
    bed_png = _png("garden/bed.png")
    if bed_png is not None:
        region = img[y0:y1, x0:x1]
        region[:] = cv2.resize(bed_png[:, :, :3], (w, h))
    else:
        img[y0:y1, x0:x1] = SKY
        soil_y = y0 + int(h * 0.78)
        cv2.rectangle(img, (x0, soil_y - int(h * 0.04)), (x1, y1), GRASS, -1)
        cv2.rectangle(img, (x0 + int(w * 0.03), soil_y), (x1 - int(w * 0.03), y1 - int(h * 0.05)),
                      SOIL, -1, cv2.LINE_AA)
    soil_y = y0 + int(h * 0.8)
    slots = config.GARDEN_PLOTS
    plot_w = (w * 0.94) / slots
    plant_h = h * 0.62
    grow = grow or {}
    p = float(np.clip(grow.get("progress", 1.0), 0.0, 1.0))
    for i in range(slots):
        bx = x0 + w * 0.03 + plot_w * (i + 0.5)
        cv2.ellipse(img, (int(bx), soil_y + int(h * 0.02)), (int(plot_w * 0.35), int(h * 0.025)),
                    0, 0, 360, SOIL_DARK, -1, cv2.LINE_AA)
        if i >= len(garden.get("plots", [])):
            continue
        plot = garden["plots"][i]
        if i == grow.get("plot") and p < 1.0:
            layer = img.copy()
            prev = grow.get("prev_stage")
            if prev is not None:
                draw_plant(img, (bx, soil_y), plant_h, plot["plant"], prev)
            draw_plant(layer, (bx, soil_y), plant_h, plot["plant"], plot["stage"],
                       scale=0.85 + 0.15 * p)
            cv2.addWeighted(layer, p, img, 1 - p, 0, img)
            draw_can(img, (int(bx - plot_w * 0.35), int(soil_y - plant_h * 0.95)),
                     plot_w * 0.5, 0, 0, tilt=35 * min(1.0, p * 2))
            if p < 0.7:
                for k in range(6):
                    dy = int((p * 3 + k * 0.17) % 1.0 * plant_h * 0.5)
                    cv2.circle(img, (int(bx - plot_w * 0.05 + (k % 3) * 6),
                                     int(soil_y - plant_h * 0.8 + dy)), 3, WATER, -1, cv2.LINE_AA)
        else:
            draw_plant(img, (bx, soil_y), plant_h, plot["plant"], plot["stage"])
    # visitors: a bee per best, a butterfly per milestone (only ever added)
    # in the sky above the flowers
    area = (x0 + w * 0.05, y0 + h * 0.05, x1 - w * 0.05, y0 + h * 0.16)
    bees = min(garden.get("bees", 0), max_extras)
    flies = min(garden.get("butterflies", 0), max_extras)
    new_bees = grow.get("new_bees", 0)
    new_flies = grow.get("new_butterflies", 0)
    for k, pos in enumerate(_extra_positions(bees, area, 11)):
        if k >= bees - new_bees and p < 1.0:
            layer = img.copy()
            draw_bee(layer, pos, h * 0.07)
            cv2.addWeighted(layer, p, img, 1 - p, 0, img)
        else:
            draw_bee(img, pos, h * 0.07)
    for k, pos in enumerate(_extra_positions(flies, area, 29)):
        if k >= flies - new_flies and p < 1.0:
            layer = img.copy()
            draw_butterfly(layer, pos, h * 0.08)
            cv2.addWeighted(layer, p, img, 1 - p, 0, img)
        else:
            draw_butterfly(img, pos, h * 0.08)
    return img


# ---------------------------------------------------------------------------
# Full screens
# ---------------------------------------------------------------------------

def _answer_row(img, text, y, x_center, labels=("Thumbs up: yes", "Thumbs down: no")):
    """Two large, calm answer boxes."""
    size, box_h, gap = 34, 74, 40
    box_w = int(max(340, max(text.measure(label, size, True) for label in labels) + 60))
    for i, (label, color) in enumerate(zip(labels, (GREEN, GREY))):
        x0 = int(x_center - box_w - gap / 2 + i * (box_w + gap))
        cv2.rectangle(img, (x0, y), (x0 + box_w, y + box_h), color, -1, cv2.LINE_AA)
        text.add(label, (x0 + box_w / 2, y + box_h * 0.64), size, WHITE, bold=True, align="center")


def card_screen(size, view, text, character=None):
    """Full-screen card: coach face, title, one message, optional icon and yes/no row."""
    w, h = size
    img = np.full((h, w, 3), BG, np.uint8)
    card = view.get("card", {})
    face_x = int(w * 0.17)
    draw_face(img, (face_x, int(h * 0.42)), int(h * 0.2), view.get("mood", "neutral"), character)
    if view.get("coach_name"):
        text.add(view["coach_name"], (face_x, int(h * 0.42 + h * 0.3)), 36, SOFT, align="center")
    x = int(w * 0.36)
    right = w - 60
    y = int(h * 0.2)
    if card.get("icon"):
        draw_icon(img, card["icon"], (right - 110, int(h * 0.24)), 170, GREEN)
        right -= 240
    y = text.add(card.get("title", ""), (x, y), 60, INK, bold=True, max_width=right - x,
                 max_lines=3) + 40
    y = text.add(card.get("message", ""), (x, y + 20), 46, INK, max_width=w - 60 - x,
                 max_lines=3) + 30
    if card.get("options"):
        for i, (label, chosen) in enumerate(card["options"]):
            cy = int(y + 30 + i * 62)
            cv2.circle(img, (x + 18, cy - 12), 14, GREEN if chosen else GREY, -1 if chosen else 3,
                       cv2.LINE_AA)
            text.add(label, (x + 50, cy), 36, INK if chosen else SOFT)
    if card.get("yes_no"):
        _answer_row(img, text, h - 190, x + (w - 60 - x) / 2, card.get("answer_labels")
                    or ("Thumbs up: yes", "Thumbs down: no"))
    can = view.get("can")
    if can and can.get("sections"):
        draw_can(img, (w - 110, h - 110), 110, can["sections"], can.get("filled", 0))
    text.add(view.get("footer", ""), (40, h - 30), 26, SOFT)
    return img


def garden_screen(size, view, text, character=None):
    """The garden, today's growth and one spoken line underneath."""
    w, h = size
    img = np.full((h, w, 3), BG, np.uint8)
    g = view.get("garden") or {}
    draw_garden(img, (40, 40, w - 40, int(h * 0.74)), g.get("state", {}), g.get("grow"))
    draw_face(img, (110, int(h * 0.86)), 60, view.get("mood", "happy"), character)
    text.add(view.get("title", ""), (210, int(h * 0.83)), 44, INK, bold=True, max_width=w - 260)
    text.add(g.get("line", ""), (210, int(h * 0.83) + 58), 36, INK, max_width=w - 260, max_lines=2)
    season = g.get("state", {}).get("season", 1)
    if season > 1:
        text.add(f"Season {season}", (w - 60, 90), 30, SOFT, align="right")
    text.add(view.get("footer", ""), (w - 40, h - 20), 24, SOFT, align="right")
    return img


def summary_screen(size, view, text, character=None):
    """One highlight, the garden as it is, and the closing message."""
    w, h = size
    img = np.full((h, w, 3), BG, np.uint8)
    draw_face(img, (120, 130), 80, view.get("mood", "happy"), character)
    text.add(view.get("title", ""), (240, 110), 56, INK, bold=True, max_width=w - 300)
    y = 190
    lines = view.get("summary_lines", [])
    if lines:
        y = text.add(lines[0], (240, y), 44, INK, bold=True, max_width=w - 300, max_lines=2) + 30
    for line in lines[1:4]:
        y = text.add(line, (240, y), 34, INK, max_width=w - 300, max_lines=2) + 18
    g = view.get("garden") or {}
    if g.get("state"):
        draw_garden(img, (240, max(y + 20, int(h * 0.5)), w - 60, h - 70), g["state"])
    text.add(view.get("footer", ""), (40, h - 25), 26, SOFT)
    return img

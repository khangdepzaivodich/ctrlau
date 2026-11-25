"""Reusable scientific-figure glyphs built from native draw.io shapes."""
import random
from figlib import *

# real SymGraphAU prior (config.py) ------------------------------------------------
EMOS = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Disgust", "Contempt", "Neutral"]
EMO3 = ["Ang", "Fea", "Hap", "Sad", "Sur", "Dis", "Con", "Neu"]
REL = {
    "Surprise": ([1, 2, 25, 26], []), "Fear": ([1, 2, 4, 7], [23, 25]), "Disgust": ([9, 10], [25]),
    "Happy": ([6, 12], [7, 14, 25]), "Sad": ([1, 7, 15], [4, 6, 17]),
    "Angry": ([4, 23, 24], [7, 17]), "Contempt": ([12, 14, 23], []), "Neutral": ([], []),
}
DISFA = [1, 2, 4, 6, 9, 12, 25, 26]


def prior(aus=DISFA, beta=0.1):
    m = []
    for a in aus:
        row = []
        for e in EMOS:
            pr, se = REL[e]
            row.append(1 - beta if a in pr else 0.5 if a in se else beta)
        m.append(row)
    return m


# ---- basic shapes ---------------------------------------------------------------------
def box(p, x, y, w, h, fill="#FFFFFF", stroke=GREY, sw=1.5, arc=10, layer=3, dashed=False,
        grad=None, gdir="south", op=None, extra="", value="", fs=13, color=INK, bold=False, align="center"):
    s = st("rounded=1", "absoluteArcSize=1", f"arcSize={arc}", "whiteSpace=wrap", "html=1",
           f"fillColor={fill}", f"strokeColor={stroke}", f"strokeWidth={sw:g}",
           f"gradientColor={grad}" if grad else "", f"gradientDirection={gdir}" if grad else "",
           "dashed=1;dashPattern=5 4" if dashed else "", f"opacity={op}" if op else "",
           f"fontSize={fs}", f"fontColor={color}", f"fontFamily={FONT}", "fontStyle=1" if bold else "",
           f"align={align}", "spacing=0", extra)
    return p.v(x, y, w, h, s, value, layer=layer)


def sq(p, x, y, w, h, fill, stroke="#FFFFFF", sw=0.8, layer=3):
    return p.v(x, y, w, h, st("rounded=0", "whiteSpace=wrap", "html=1", f"fillColor={fill}",
                              f"strokeColor={stroke}", f"strokeWidth={sw:g}"), layer=layer)


def ell(p, cx, cy, d, fill, stroke, sw=1.5, layer=3, value="", fs=11, color=INK, bold=False,
        extra="", dx=None, grad=None):
    dx = d if dx is None else dx
    s = st("ellipse", "whiteSpace=nowrap", "html=1", f"fillColor={fill}", f"strokeColor={stroke}",
           f"strokeWidth={sw:g}", f"fontSize={fs}", f"fontColor={color}", f"fontFamily={FONT}",
           "fontStyle=1" if bold else "", "spacing=0", f"gradientColor={grad}" if grad else "",
           "gradientDirection=south" if grad else "", extra)
    return p.v(cx - dx / 2, cy - d / 2, dx, d, s, value, layer)


def trap(p, cx, cy, W, H, fill, stroke, narrow="right", inset=None, sw=1.5, grad=None, layer=3,
         extra="", arc=6):
    """Trapezoid occupying final box W x H centred at (cx, cy); `inset` shortens the narrow end per side."""
    inset = inset if inset is not None else min(W, H) * 0.2
    if narrow in ("right", "left"):
        rot, w, h = (90 if narrow == "right" else 270), H, W
    else:
        rot, w, h = (0 if narrow == "top" else 180), W, H
    s = st("shape=trapezoid", "perimeter=trapezoidPerimeter", "whiteSpace=wrap", "html=1", "fixedSize=1",
           f"size={inset:g}", "rounded=1", f"arcSize={arc}", f"fillColor={fill}", f"strokeColor={stroke}",
           f"strokeWidth={sw:g}", f"gradientColor={grad}" if grad else "",
           "gradientDirection=north" if grad else "", f"rotation={rot}" if rot else "", extra)
    return p.v(cx - w / 2, cy - h / 2, w, h, s, layer=layer)


def cube(p, x, y, w, h, d, fill, stroke, sw=1.2, layer=3, grad=None):
    s = st("shape=cube", "whiteSpace=wrap", "html=1", "boundedLbl=1", "backgroundOutline=1",
           "darkOpacity=0.06", "darkOpacity2=0.16", f"size={d}", f"fillColor={fill}", f"strokeColor={stroke}",
           f"strokeWidth={sw:g}", f"gradientColor={grad}" if grad else "",
           "gradientDirection=north" if grad else "")
    return p.v(x, y, w, h, s, layer=layer)


def dots(p, cx, cy, vertical=True, color=FAINT, n=3, gap=8, d=4.5, layer=3):
    for i in range(n):
        o = (i - (n - 1) / 2) * gap
        ell(p, cx + (0 if vertical else o), cy + (o if vertical else 0), d, color, color, sw=0.5, layer=layer)


def badge(p, cx, cy, text, color, d=24, fs=13, layer=6):
    return ell(p, cx, cy, d, color, "#FFFFFF", sw=1.5, value=text, fs=fs, color="#FFFFFF", bold=True, layer=layer)


def frozen(p, cx, cy, d=22, layer=6):
    return ell(p, cx, cy, d, "#E6F3FC", "#6FA8D6", sw=1.3, value="\u2744", fs=12, color="#2F7DB8", layer=layer)


def op(p, cx, cy, sym, d=26, fill="#FFFFFF", stroke=LINE, fs=15, layer=4, color=INK, sw=1.6):
    return ell(p, cx, cy, d, fill, stroke, sw=sw, value=sym, fs=fs, color=color, layer=layer)


def pill(p, cx, cy, html, color=CRIM, fill=CRIM_L, fs=13, h=26, w=None, layer=5, stroke=None, bold=False):
    n, _ = plain_len(html)
    w = w or n * fs * 0.6 + 22
    s = st("rounded=1", "absoluteArcSize=1", f"arcSize={h}", "html=1", "whiteSpace=nowrap", f"fillColor={fill}",
           f"strokeColor={stroke or mix('#FFFFFF', color, 0.55)}", "strokeWidth=1.2", f"fontSize={fs}",
           f"fontColor={color}", f"fontFamily={FONT}", "fontStyle=1" if bold else "", "spacing=0")
    return p.v(cx - w / 2, cy - h / 2, w, h, s, html, layer=layer)


def panel(p, x, y, w, h, color, title, num=None, tag=None, fill=None, arc=22, dashed=False, chip_x=26, fs=16):
    """Soft tinted panel with a numbered title chip straddling its top border."""
    fill = fill or mix("#FFFFFF", color, 0.06)
    stroke = mix("#FFFFFF", color, 0.5)
    p.v(x, y, w, h, st("rounded=1", "absoluteArcSize=1", f"arcSize={arc}", "html=1", f"fillColor={fill}",
                       f"strokeColor={stroke}", "strokeWidth=1.8",
                       "dashed=1;dashPattern=7 5" if dashed else ""), layer=0)
    n, _ = plain_len(title)
    lead = 40 if num else 18
    cw = n * fs * 0.56 + lead + 18
    p.v(x + chip_x, y - 17, cw, 34,
        st("rounded=1", "absoluteArcSize=1", "arcSize=17", "html=1", f"fillColor={color}",
           "strokeColor=#FFFFFF", "strokeWidth=2.5"), layer=1)
    if num:
        ell(p, x + chip_x + 20, y, 24, "#FFFFFF", "#FFFFFF", sw=0.5, value=str(num), fs=14, color=color,
            bold=True, layer=6)
    p.t(x + chip_x + lead, y, title, fs=fs, color="#FFFFFF", bold=True, align="left", w=cw - lead, layer=6)
    if tag:
        pill(p, x + w - 24 - (len(tag) * 7.2 + 24) / 2, y, tag, color=color, fill="#FFFFFF", fs=12.5, h=28,
             w=len(tag) * 7.2 + 24, bold=True, layer=6, stroke=color)
    return cw


def card(p, x, y, w, h, color, fill="#FFFFFF", arc=14, sw=1.5, dashed=False, layer=1):
    return box(p, x, y, w, h, fill=fill, stroke=mix("#FFFFFF", color, 0.6), sw=sw, arc=arc, layer=layer,
               dashed=dashed)


# ---- tensors / data glyphs ------------------------------------------------------------------
def vec(p, x, y, w, h, n, color, seed=0, layer=3, frame=None, lo=0.25):
    rnd = random.Random(seed)
    ch = h / n
    with p.group(x, y, w, h, layer=layer):
        for i in range(n):
            sq(p, x, y + i * ch, w, ch, mix("#FFFFFF", color, rnd.uniform(lo, 1.0)), layer=layer)
        p.v(x, y, w, h, st("rounded=1", "absoluteArcSize=1", "arcSize=4", "html=1", "fillColor=none",
                           f"strokeColor={frame or color}", "strokeWidth=1.4"), layer=layer)


def hvec(p, x, y, w, h, n, color, seed=0, layer=3, frame=None, lo=0.25, active=None):
    """Horizontal strip of n cells; `active` = set of indices drawn solid dark."""
    rnd = random.Random(seed)
    cw = w / n
    with p.group(x, y, w, h, layer=layer):
        for i in range(n):
            if active is not None:
                f = color if i in active else "#FFFFFF"
            else:
                f = mix("#FFFFFF", color, rnd.uniform(lo, 1.0))
            sq(p, x + i * cw, y, cw, h, f, layer=layer)
        p.v(x, y, w, h, st("rounded=1", "absoluteArcSize=1", "arcSize=4", "html=1", "fillColor=none",
                           f"strokeColor={frame or color}", "strokeWidth=1.4"), layer=layer)


def rows(p, x, y, w, h, n, color, seed=0, layer=3, pad=6, gap=4, lo=0.3):
    """N x D embedding matrix: framed, horizontal bars (one per node)."""
    rnd = random.Random(seed)
    bh = (h - 2 * pad - (n - 1) * gap) / n
    with p.group(x, y, w, h, layer=layer):
        box(p, x, y, w, h, fill="#FFFFFF", stroke=color, sw=1.5, arc=6, layer=layer)
        for i in range(n):
            segs = 6
            sw_ = (w - 2 * pad) / segs
            for j in range(segs):
                sq(p, x + pad + j * sw_, y + pad + i * (bh + gap), sw_, bh,
                   mix("#FFFFFF", color, rnd.uniform(lo, 1.0)), stroke="#FFFFFF", sw=0.6, layer=layer)


def heat(p, x, y, cell, vals, color, lo="#FAF8FC", gap=1.0, frame=None, layer=3, signed=None, fw=1.3):
    r, c = len(vals), len(vals[0])
    W, H = c * cell, r * cell
    with p.group(x, y, W, H, layer=layer):
        for i in range(r):
            for j in range(c):
                v = vals[i][j]
                if signed:
                    pos, neg = signed
                    f = mix(lo, pos, min(1, v)) if v >= 0 else mix(lo, neg, min(1, -v))
                else:
                    f = mix(lo, color, v)
                sq(p, x + j * cell, y + i * cell, cell, cell, f, stroke="#FFFFFF", sw=gap, layer=layer)
        p.v(x, y, W, H, st("rounded=0", "html=1", "fillColor=none", f"strokeColor={frame or color}",
                           f"strokeWidth={fw:g}"), layer=layer)
    return W, H


def barchart(p, x, y, w, h, vals, color, hl=(), hlc=None, layer=3, base="#9AA5B1"):
    n = len(vals)
    bw = w / (n * 1.55)
    gap = (w - n * bw) / (n + 1)
    with p.group(x, y, w, h, layer=layer):
        for i, v in enumerate(vals):
            bh = max(2, v * (h - 4))
            f = (hlc or color) if i in hl else mix("#FFFFFF", color, 0.45)
            sq(p, x + gap + i * (bw + gap), y + h - bh, bw, bh, f,
               stroke=mix("#FFFFFF", color, 0.9) if i in hl else "#FFFFFF", sw=0.8, layer=layer)
        p.e([(x - 2, y + h), (x + w + 2, y + h)], color=base, w=1.4, head=None, layer=layer + 1, rounded=False)
    return


def face(p, x, y, s, layer=3):
    """Photo card stack with a simple smiling portrait."""
    box(p, x + 9, y - 9, s, s, fill="#F3F5F7", stroke="#B7C0CA", sw=1.4, arc=8, layer=layer)
    box(p, x, y, s, s, fill="#FFFFFF", stroke="#6B7785", sw=1.8, arc=8, layer=layer)
    cx, cy = x + s / 2, y + s / 2
    trap(p, cx, y + s * 0.80, s * 0.70, s * 0.30, "#8FA6C0", "#5F7896", narrow="top", inset=s * 0.12, sw=1.2,
         layer=layer, arc=30)
    ell(p, cx, cy - s * 0.06, s * 0.46, "#F1D2B2", "#B88A63", sw=1.3, layer=layer)
    ell(p, cx - s * 0.085, cy - s * 0.09, s * 0.05, "#3A3F47", "#3A3F47", sw=0.5, layer=layer)
    ell(p, cx + s * 0.085, cy - s * 0.09, s * 0.05, "#3A3F47", "#3A3F47", sw=0.5, layer=layer)
    p.e([(cx - s * 0.08, cy + s * 0.03), (cx, cy + s * 0.085), (cx + s * 0.08, cy + s * 0.03)],
        color="#8A5A3C", w=1.4, head=None, curved=True, layer=layer + 1)


def tokens(p, cx, cy, n=7, cell=10.5, gap=1.6, color=BLUE, layer=3, seed=3):
    rnd = random.Random(seed)
    W = n * cell + (n - 1) * gap
    x0, y0 = cx - W / 2, cy - W / 2
    for k in (2, 1):
        box(p, x0 + 6 * k, y0 - 6 * k, W, W, fill="#FFFFFF", stroke=mix("#FFFFFF", color, 0.7), sw=1.2, arc=3,
            layer=layer)
    with p.group(x0, y0, W, W, layer=layer):
        box(p, x0 - 3, y0 - 3, W + 6, W + 6, fill="#FFFFFF", stroke=color, sw=1.5, arc=4, layer=layer)
        for i in range(n):
            for j in range(n):
                sq(p, x0 + j * (cell + gap), y0 + i * (cell + gap), cell, cell,
                   mix("#FFFFFF", color, rnd.uniform(0.25, 0.95)), stroke="none", layer=layer)
    return W


def node(p, cx, cy, label, kind="au", d=32, fs=11.5, layer=3):
    if kind == "au":
        fill, stroke, col = TERRA_L, TERRA, "#7A2F20"
    else:
        fill, stroke, col = "#D7E3F7", "#4F73B5", "#233E70"
    return ell(p, cx, cy, d, fill, stroke, sw=1.6, value=label, fs=fs, color=col, bold=True, layer=layer,
               grad=mix(fill, "#FFFFFF", 0.55))


def brace(p, x, y, w, h, color=GREY, rot=0, size=0.5, layer=3):
    s = st("shape=curlyBracket", "rounded=1", f"size={size}", "html=1", "fillColor=none", f"strokeColor={color}",
           "strokeWidth=1.6", f"rotation={rot}" if rot else "")
    return p.v(x, y, w, h, s, layer=layer)


def cap(p, cx, cy, text, fs=12, color=MUTE, **kw):
    return p.t(cx, cy, text, fs=fs, color=color, **kw)

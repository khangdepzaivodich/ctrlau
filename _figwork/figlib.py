"""Tiny draw.io scene builder: native vector shapes, groups, explicit-polyline arrows."""
import math
import re
from contextlib import contextmanager
from xml.sax.saxutils import escape as _esc

FONT = "Helvetica"
SERIF = "Times New Roman"

# ---- palette ---------------------------------------------------------------
INK = "#27313D"
MUTE = "#6B7785"
FAINT = "#9AA5B1"
LINE = "#44546A"

BLUE, BLUE_L, BLUE_M = "#3B6EA8", "#DDE9F7", "#A5C0E3"
TEAL, TEAL_L, TEAL_M = "#2A8C7D", "#D8F0EB", "#93CDC1"
PURP, PURP_L, PURP_M = "#7656A6", "#EAE1F4", "#BBA5D8"
TERRA, TERRA_L = "#BF5F4A", "#F4D5CB"
AMBER, AMBER_L, AMBER_M = "#D28A12", "#FCEBC9", "#F0C36B"
CRIM, CRIM_L, CRIM_M = "#B23A5B", "#F7DEE5", "#E3A0B3"
GREEN, GREEN_L = "#4E9A52", "#DCF0DC"
GREY, GREY_L, GREY_M = "#7B8794", "#EEF1F4", "#C9D1D9"
SLATE, SLATE_L = "#5F7896", "#8FA6C0"


def esc(s):
    return _esc(str(s), {'"': "&quot;"})


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb2hex(c):
    return "#%02X%02X%02X" % tuple(max(0, min(255, int(round(v)))) for v in c)


def mix(c1, c2, t):
    a, b = hex2rgb(c1), hex2rgb(c2)
    return rgb2hex([a[i] + (b[i] - a[i]) * t for i in range(3)])


def st(*parts, **kw):
    """Build a draw.io style string. Bare strings are copied, kwargs become k=v."""
    out = [p.strip(";") for p in parts if p]
    for k, v in kw.items():
        if v is None:
            continue
        if isinstance(v, bool):
            v = int(v)
        if isinstance(v, float):
            v = f"{v:g}"
        out.append(f"{k}={v}")
    return ";".join(out) + ";"


# ---- html helpers for math-y labels ----------------------------------------
def I(s):
    return f'<i style="font-family:{SERIF}">{s}</i>'


def sub(s):
    return f"<sub>{s}</sub>"


def sup(s):
    return f"<sup>{s}</sup>"


def var(base, s=None, p=None):
    r = I(base)
    if s:
        r += sub(s)
    if p:
        r += sup(p)
    return r


def Lc(s=None, p=None):
    """script L loss symbol with optional subscript."""
    return f'<span style="font-family:{SERIF}">&#8466;</span>' + (sub(s) if s else "") + (sup(p) if p else "")


def plain_len(html):
    t = re.sub(r"<br\s*/?>", "\n", html)
    t = re.sub(r"<[^>]+>", "", t)
    t = t.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    lines = t.split("\n")
    return max(len(x) for x in lines), len(lines)


class Page:
    NL = 8

    def __init__(self, name, w, h, pid=None):
        self.name, self.w, self.h = name, w, h
        self.pid = pid or re.sub(r"\W+", "_", name).lower()
        self.layers = [[] for _ in range(self.NL)]
        self._k = 1
        self.gstack = []
        # white page-sized backdrop: fixes the export bounds and the background colour
        self.v(0, 0, w, h, "rounded=0;html=1;fillColor=#FFFFFF;strokeColor=none;pointerEvents=0;movable=0;resizable=0;"
                           "rotatable=0;deletable=0;editable=0;connectable=0;", layer=0)

    # -- ids / emit -----------------------------------------------------------
    def nid(self):
        self._k += 1
        return f"n{self._k}"

    def _emit(self, layer, xml):
        if self.gstack:
            self.gstack[-1]["buf"].append(xml)
        else:
            self.layers[layer].append(xml)

    # -- primitives -----------------------------------------------------------
    def v(self, x, y, w, h, style, value="", layer=3):
        i = self.nid()
        if self.gstack:
            g = self.gstack[-1]
            rx, ry, parent = x - g["ox"], y - g["oy"], g["id"]
        else:
            rx, ry, parent = x, y, "1"
        self._emit(
            layer,
            f'<mxCell id="{i}" value="{esc(value)}" style="{style}" vertex="1" parent="{parent}">'
            f'<mxGeometry x="{rx:.2f}" y="{ry:.2f}" width="{w:.2f}" height="{h:.2f}" as="geometry"/></mxCell>',
        )
        return i

    @contextmanager
    def group(self, x, y, w, h, layer=3):
        gid = self.nid()
        self.gstack.append({"id": gid, "ox": x, "oy": y, "buf": []})
        try:
            yield gid
        finally:
            g = self.gstack.pop()
            if self.gstack:
                p = self.gstack[-1]
                rx, ry, parent = x - p["ox"], y - p["oy"], p["id"]
            else:
                rx, ry, parent = x, y, "1"
            self._emit(
                layer,
                f'<mxCell id="{gid}" value="" style="group;" vertex="1" connectable="0" parent="{parent}">'
                f'<mxGeometry x="{rx:.2f}" y="{ry:.2f}" width="{w:.2f}" height="{h:.2f}" as="geometry"/></mxCell>',
            )
            for c in g["buf"]:
                self._emit(layer, c)

    def e(self, pts, color=LINE, w=2.0, dashed=False, head="block", tail=None, curved=False,
          rounded=True, layer=2, arc=10, hs=None, ts=None, opacity=None, dash="4 3", extra=""):
        """Polyline arrow through absolute points. head/tail: block|classic|open|none|oval|diamond."""
        i = self.nid()
        kw = {}
        if head in (None, "none"):
            kw["endArrow"] = "none"
        else:
            kw["endArrow"] = head
            kw["endFill"] = 1
            kw["endSize"] = hs if hs is not None else 5
        if tail in (None, "none"):
            kw["startArrow"] = "none"
        else:
            kw["startArrow"] = tail
            kw["startFill"] = 1
            kw["startSize"] = ts if ts is not None else 5
        s = st("edgeStyle=none", "html=1", f"rounded={int(rounded)}", f"arcSize={arc}",
               f"curved={int(curved)}", f"strokeWidth={w:g}", f"strokeColor={color}",
               f"dashed={int(dashed)}", f"dashPattern={dash}" if dashed else "",
               f"opacity={opacity}" if opacity is not None else "", extra, **kw)
        mid = "".join(f'<mxPoint x="{px:.2f}" y="{py:.2f}"/>' for px, py in pts[1:-1])
        arr = f'<Array as="points">{mid}</Array>' if mid else ""
        self.layers[layer].append(
            f'<mxCell id="{i}" value="" style="{s}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
            f'<mxPoint x="{pts[0][0]:.2f}" y="{pts[0][1]:.2f}" as="sourcePoint"/>'
            f'<mxPoint x="{pts[-1][0]:.2f}" y="{pts[-1][1]:.2f}" as="targetPoint"/>{arr}</mxGeometry></mxCell>'
        )
        return i

    # -- text -----------------------------------------------------------------
    def t(self, cx, cy, html, fs=13, color=INK, bold=False, italic=False, family=FONT,
          align="center", w=None, h=None, bg=None, layer=5, rot=0, lh=None, pad=None):
        """Text centered at (cx, cy); if align=left/right the (cx,cy) is the left/right-middle anchor."""
        n, lines = plain_len(html)
        tw = w if w is not None else max(24, n * fs * 0.62 + 10)
        th = h if h is not None else lines * fs * 1.32 + 4
        if align == "center":
            x = cx - tw / 2
        elif align == "left":
            x = cx
        else:
            x = cx - tw
        y = cy - th / 2
        fstyle = (1 if bold else 0) | (2 if italic else 0)
        s = st("text", "html=1", "whiteSpace=nowrap", "spacing=0", f"align={align}", "verticalAlign=middle",
               f"fontSize={fs}", f"fontColor={color}", f"fontFamily={family}",
               f"fontStyle={fstyle}" if fstyle else "",
               f"labelBackgroundColor={bg}" if False else "",
               f"fillColor={bg}" if bg else "fillColor=none", "strokeColor=none",
               f"rotation={rot}" if rot else "",
               f"spacingLeft={pad}" if pad else "", f"spacingRight={pad}" if pad else "",
               f"lineHeight={lh}" if lh else "")
        return self.v(x, y, tw, th, s, html, layer=layer)

    # -- document ---------------------------------------------------------------
    def xml(self):
        cells = "\n        ".join(c for L in self.layers for c in L)
        return (
            f'<diagram id="{self.pid}" name="{esc(self.name)}">\n'
            f'  <mxGraphModel grid="0" gridSize="10" guides="1" tooltips="1" connect="1" '
            f'arrows="1" fold="1" page="1" pageScale="1" pageWidth="{self.w}" pageHeight="{self.h}" '
            f'background="#FFFFFF" math="0" shadow="0">\n'
            f'    <root>\n      <mxCell id="0"/>\n      <mxCell id="1" parent="0"/>\n        {cells}\n    </root>\n'
            f"  </mxGraphModel>\n</diagram>\n"
        )


def write_drawio(path, pages):
    body = "".join(p.xml() for p in pages)
    with open(path, "w", encoding="utf-8") as f:
        f.write('<mxfile host="app.diagrams.net" agent="ctrlau-figure" version="24.0.0">\n' + body + "</mxfile>\n")

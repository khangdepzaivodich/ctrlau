"""Page 2 - graph reasoning layer in detail."""
import math
import random
from shapes import *
from page1 import rmat, smat, edge_between, minigraph, ticks, AHAT

W, H = 2400, 1090
PBG = mix("#FFFFFF", PURP, 0.06)


def proj(p, cx, cy, label, color=PURP):
    trap(p, cx, cy, 34, 70, mix("#FFFFFF", color, 0.45), color, narrow="right", inset=15,
         grad=mix("#FFFFFF", color, 0.15))
    p.t(cx, cy - 50, label, fs=13, color=color)


def mat(p, cx, cy, label, color, seed, w=46, h=72, n=5, lab_dy=-50):
    rows(p, cx - w / 2, cy - h / 2, w, h, n, color, seed=seed, pad=4, gap=2)
    p.t(cx, cy + lab_dy, label, fs=14, color=color)


def anatomy(p):
    panel(p, 30, 130, 1580, 585, PURP, "Dense dynamic graph convolution", num=1, tag="shared by both graphs")
    cy = 405
    # ----- dashed "x2 layers" frame
    box(p, 206, 150, 1100, 545, fill="none", stroke=mix("#FFFFFF", PURP, 0.55), sw=1.6, arc=16, dashed=True, layer=1)
    pill(p, 1262, 150, "\u00d7 2 layers", color=PURP, fill="#FFFFFF", fs=12.5, h=26, w=96, bold=True, stroke=PURP)

    # ----- input X
    rows(p, 76, cy - 80, 84, 160, 8, BLUE, seed=3, pad=6, gap=3)
    p.t(118, cy - 126, "node embeddings", fs=11.5, color=MUTE)
    p.t(118, cy - 108, var("X") + "  (N \u00d7 D)", fs=15, color=BLUE)

    # ----- Q K V projections
    ys = {"q": 290, "k": 405, "v": 530}
    for key, (lab, seed) in {"q": ("W<sub>q</sub>", 31), "k": ("W<sub>k</sub>", 32), "v": ("W<sub>v</sub>", 33)}.items():
        y = ys[key]
        p.e([(160, cy), (186, cy), (186, y), (218, y)], color=PURP_M if False else LINE, w=2.0, arc=12)
        proj(p, 236, y, I(lab))
        p.e([(254, y), (290, y)], color=PURP, w=1.8)
        mat(p, 314, y, I(key.upper()), PURP, seed)

    # ----- affinity branch
    op(p, 410, 348, "\u00d7", d=28, fs=17, color=PURP, stroke=PURP)
    p.e([(338, ys["q"]), (410, ys["q"]), (410, 334)], color=PURP, w=1.8, arc=10)
    p.e([(338, ys["k"]), (410, ys["k"]), (410, 362)], color=PURP, w=1.8, arc=10)
    p.e([(424, 348), (460, 348)], color=PURP, w=1.8)
    op(p, 474, 348, "\u03c3", d=28, fs=15, color=PURP, stroke=PURP)
    p.e([(488, 348), (520, 348)], color=PURP, w=1.8)
    heat(p, 522, 318, 10, rmat(6, 6, 5, lo=0.1, hi=0.95), PURP, lo="#FAF8FC", gap=0.8)
    p.t(552, 396, var("B") + I("(x)"), fs=14, color=PURP)
    p.t(552, 414, "per-image affinity", fs=11, color=MUTE)
    p.t(552, 430, "\u03c3(QK<sup>T</sup>/\u221ad) \u2208 (0, 1)", fs=11, color=MUTE)

    # ----- global branch
    heat(p, 412, 160, 10, smat(6, 6, 3), PURP, lo="#FAF8FC", signed=(PURP, AMBER), gap=0.8)
    p.t(442, 238, var("G"), fs=14, color=PURP)
    pill(p, 442, 256, "learnable", color=PURP, fill=PURP_L, fs=10.5, h=20, w=62, stroke=PURP)
    p.e([(474, 190), (500, 190)], color=PURP, w=1.8)
    op(p, 516, 190, "tanh", d=34, fs=10, color=PURP, stroke=PURP)
    p.e([(533, 190), (562, 190)], color=PURP, w=1.8)
    heat(p, 564, 160, 10, smat(6, 6, 3), PURP, lo="#FAF8FC", signed=(PURP, AMBER), gap=0.8)
    p.t(594, 238, var("A"), fs=14, color=PURP)
    p.t(594, 256, "signed global edges", fs=11, color=MUTE)

    # ----- combine
    op(p, 704, 290, "\u2299", d=28, fs=16, color=PURP, stroke=PURP)
    p.e([(626, 190), (704, 190), (704, 276)], color=PURP, w=1.8, arc=10)
    p.e([(584, 348), (704, 348), (704, 304)], color=PURP, w=1.8, arc=10)
    p.e([(718, 290), (752, 290)], color=PURP, w=1.8)
    pill(p, 786, 290, "\u00f7 \u03a3|\u00b7|", color=PURP, fill=PURP_L, fs=12, h=26, w=64, stroke=PURP)
    p.e([(819, 290), (846, 290)], color=PURP, w=1.8)
    heat(p, 848, 260, 10, [[abs(v) if (i + j) % 2 else v for j, v in enumerate(r)] for i, r in enumerate(smat(6, 6, 8, 0.3))],
         PURP, lo="#FAF8FC", signed=(PURP, AMBER), gap=0.8)
    p.t(878, 338, AHAT, fs=15, color=PURP)
    p.t(878, 356, "row-normalised,", fs=11, color=MUTE)
    p.t(878, 371, "sign kept", fs=11, color=MUTE)

    # ----- message passing
    op(p, 996, 348, "\u00d7", d=28, fs=17, color=PURP, stroke=PURP)
    p.e([(912, 290), (996, 290), (996, 334)], color=PURP, w=1.8, arc=10)
    p.e([(338, ys["v"]), (996, ys["v"]), (996, 362)], color=PURP, w=1.8, arc=10)
    p.e([(1010, 348), (1040, 348)], color=PURP, w=1.8)
    proj(p, 1058, 348, I("W<sub>o</sub>"))
    p.e([(1076, 348), (1100, 348)], color=PURP, w=1.8)
    op(p, 1116, 348, "ELU", d=32, fs=10, color=PURP, stroke=PURP)
    p.e([(1132, 348), (1176, 348)], color=PURP, w=1.8)
    op(p, 1192, 348, "+", d=28, fs=18, color=PURP, stroke=PURP)
    p.e([(1206, 348), (1240, 348)], color=PURP, w=1.8)
    mat(p, 1264, 348, var("X", "", "(\u2113)"), PURP, 41, w=46, h=100, n=6, lab_dy=-68)

    # inner residual
    p.e([(118, cy + 80), (118, 600), (1192, 600), (1192, 362)], color=BLUE, w=2.0, arc=14)
    p.t(660, 600, "inner residual   " + var("W", "r") + var("x") + "  (identity if dimensions match)", fs=12, color=BLUE,
        bg=PBG, w=310)

    # ----- outer projection + residual
    p.e([(1310, 348), (1346, 348)], color=PURP, w=2.0)
    proj(p, 1366, 348, I("W<sub>p</sub>"))
    p.e([(1384, 348), (1424, 348)], color=PURP, w=2.0)
    op(p, 1440, 348, "+", d=28, fs=18, color=PURP, stroke=PURP)
    p.e([(1454, 348), (1490, 348)], color=PURP, w=2.4)
    mat(p, 1516, 348, var("U"), PURP, 51, w=46, h=100, n=6, lab_dy=-68)
    p.e([(96, cy + 80), (96, 650), (1440, 650), (1440, 362)], color=BLUE, w=2.0, arc=14)
    p.t(900, 650, "outer residual (identity)", fs=12, color=BLUE, bg=PBG, w=170)
    p.t(1516, 418, "updated", fs=11, color=MUTE)
    p.t(1516, 433, "node features", fs=11, color=MUTE)


def block_matrix(p, x, y, cell=30, n_au=5, n_e=4):
    N = n_au + n_e
    rnd = random.Random(8)
    with p.group(x, y, N * cell, N * cell):
        for i in range(N):
            for j in range(N):
                learn = i < n_au and j >= n_au
                f = mix("#FFFFFF", PURP, rnd.uniform(0.25, 0.95)) if learn else "#F4F5F8"
                sq(p, x + j * cell, y + i * cell, cell, cell, f, stroke="#FFFFFF", sw=1.4)
        # group separators
        p.v(x, y, N * cell, N * cell, st("rounded=0", "html=1", "fillColor=none", f"strokeColor={GREY}", "strokeWidth=1.4"))
    # emphasise learnable block
    p.v(x + n_au * cell, y, n_e * cell, n_au * cell,
        st("rounded=0", "html=1", "fillColor=none", f"strokeColor={PURP}", "strokeWidth=3"), layer=4)
    # separators between groups
    p.e([(x + n_au * cell, y), (x + n_au * cell, y + N * cell)], color=GREY, w=1.6, head=None, rounded=False, layer=4)
    p.e([(x, y + n_au * cell), (x + N * cell, y + n_au * cell)], color=GREY, w=1.6, head=None, rounded=False, layer=4)
    for (i0, j0, n_i, n_j) in [(0, 0, n_au, n_au), (n_au, 0, n_e, n_au), (n_au, n_au, n_e, n_e)]:
        p.t(x + (j0 + n_j / 2) * cell, y + (i0 + n_i / 2) * cell, "0", fs=16, color=FAINT)
    p.t(x + (n_au + n_e / 2) * cell, y + n_au / 2 * cell, var("A", "AE"), fs=20, color="#FFFFFF" if False else PURP, bold=False)


def topology(p):
    panel(p, 1640, 130, 730, 585, PURP, "AU \u2192 Expression: one-way topology", num=2)
    cell, n_au, n_e = 30, 5, 4
    x, y = 1745, 290
    block_matrix(p, x, y, cell, n_au, n_e)
    N = n_au + n_e
    # axis annotations
    p.t(x + N * cell / 2, y - 62, "target (receiver)", fs=12, color=MUTE)
    p.t(x + n_au * cell / 2, y - 24, "AU", fs=12.5, bold=True, color=TERRA)
    p.t(x + (n_au + n_e / 2) * cell, y - 24, "Expression", fs=12.5, bold=True, color="#4F73B5")
    p.e([(x + 2, y - 10), (x + n_au * cell - 2, y - 10)], color=TERRA, w=2.0, head=None, rounded=False)
    p.e([(x + n_au * cell + 2, y - 10), (x + N * cell - 2, y - 10)], color="#4F73B5", w=2.0, head=None, rounded=False)
    p.t(x - 62, y + N * cell / 2, "source (sender)", fs=12, color=MUTE, rot=270, w=140)
    p.t(x - 22, y + n_au * cell / 2, "AU", fs=12.5, bold=True, color=TERRA, w=30)
    p.t(x - 22, y + (n_au + n_e / 2) * cell, "Expr", fs=12.5, bold=True, color="#4F73B5", w=34)
    p.e([(x - 10, y + 2), (x - 10, y + n_au * cell - 2)], color=TERRA, w=2.0, head=None, rounded=False)
    p.e([(x - 10, y + n_au * cell + 2), (x - 10, y + N * cell - 2)], color="#4F73B5", w=2.0, head=None, rounded=False)
    p.t(x + N * cell / 2, y + N * cell + 28, "learnable block only: " + var("A", "AE") + " (N<sub>AU</sub> \u00d7 8)", fs=12.5,
        color=PURP)
    p.t(x + N * cell / 2, y + N * cell + 48, "all other entries are fixed at 0", fs=11.5, color=MUTE)

    # bipartite message glyph
    gx_a, gx_e = 2145, 2310
    for k in range(4):
        ya_ = 300 + k * 52
        for j in range(3):
            ye_ = 318 + j * 66
            s, e = edge_between((gx_a, ya_), (gx_e, ye_), 15, 16)
            strong = (k, j) in [(0, 2), (1, 2), (2, 0), (3, 0), (1, 1)]
            if strong:
                p.e([s, e], color=PURP, w=2.2, hs=3.6, layer=2)
            else:
                p.e([s, e], color=PURP_M, w=0.8, head=None, opacity=50, layer=2)
    for k, lab in enumerate(["1", "6", "12", "25"]):
        node(p, gx_a, 300 + k * 52, lab, "au", d=30, fs=11)
    for j, lab in enumerate(["Ang", "Hap", "Sur"]):
        node(p, gx_e, 318 + j * 66, lab, "ex", d=32, fs=10)
    p.t(gx_a, 252, "senders", fs=11.5, color=MUTE)
    p.t(gx_e, 252, "receivers", fs=11.5, color=MUTE)
    p.t((gx_a + gx_e) / 2 + 10, 520, "after the receiver-first transpose only", fs=11.5, color=MUTE)
    p.t((gx_a + gx_e) / 2 + 10, 537, "expression nodes receive messages;", fs=11.5, color=MUTE)
    p.t((gx_a + gx_e) / 2 + 10, 554, "AU nodes keep their residual only.", fs=11.5, color=MUTE)
    pill(p, (gx_a + gx_e) / 2 + 10, 610, "no Expr \u2192 AU feedback", color="#2E6B31", fill=GREEN_L, fs=12.5, h=28, w=200,
         bold=True, stroke=GREEN)
    p.t(2005, 640, "same layer, applied to the node set [ U<sub>a</sub> ; V<sub>e</sub> ]", fs=12, color=MUTE, italic=True,
        align="center", w=320) if False else None


def tile(p, x, y, w, h, color, title, lines, glyph):
    card(p, x, y, w, h, color, arc=12)
    glyph(x + 20, y + 40)
    tx = x + 150
    p.t(tx, y + 40, title, fs=17, bold=True, color=color, align="left")
    for i, ln in enumerate(lines):
        p.t(tx, y + 76 + i * 25, ln, fs=13, color=INK if i == 0 else MUTE, align="left")


def losses(p):
    panel(p, 30, 770, 2340, 270, GREY, "Supervision of the graph stage", num=3, tag="weights from train.py")
    tw, th, gap, y = 440, 196, 35, 818
    xs = [48 + k * (tw + gap) for k in range(5)]

    def g_bars(color, hl):
        def f(x, y):
            barchart(p, x + 6, y + 22, 90, 78, [0.2, 0.75, 0.15, 0.9, 0.3, 0.1], color, hl=hl, hlc=color)
        return f

    def g_two(x, y):
        barchart(p, x, y + 30, 50, 60, [0.2, 0.8, 0.25, 0.6], BLUE, hl=(1, 3), hlc=BLUE)
        barchart(p, x + 58, y + 30, 50, 60, [0.1, 0.1, 0.85, 0.1], TEAL, hl=(2,), hlc=TEAL)

    def g_dag(x, y):
        pts = [(x + 20, y + 90), (x + 60, y + 30), (x + 100, y + 90)]
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            s, e = edge_between(pts[a], pts[b], 13, 13)
            p.e([s, e], color=PURP, w=2.0, hs=4)
        for q in pts:
            ell(p, q[0], q[1], 24, PURP_L, PURP, sw=1.5)
        s, e = edge_between(pts[2], pts[0], 13, 13)
        p.e([(pts[2][0] - 14, pts[2][1] + 22), (pts[0][0] + 14, pts[0][1] + 22)], color=CRIM, w=1.6, dashed=True, hs=4)
        p.t(x + 60, y + 128, "\u2715 no cycles", fs=11, color=CRIM)

    def g_facs(x, y):
        node(p, x + 22, y + 60, "26", "au", d=34, fs=12)
        node(p, x + 98, y + 60, "25", "au", d=34, fs=12)
        s, e = edge_between((x + 22, y + 60), (x + 98, y + 60), 17, 17)
        p.e([s, e], color=PURP, w=2.4, hs=4)
        p.t(x + 60, y + 38, "implies", fs=11, color=MUTE)
        p.t(x + 60, y + 100, "jaw drop \u21d2 lips part", fs=10.5, color=MUTE, w=140)

    def g_cau(x, y):
        node(p, x + 20, y + 60, "A", "au", d=34, fs=12)
        node(p, x + 100, y + 60, "B", "au", d=34, fs=12)
        p.e([(x + 38, y + 52), (x + 82, y + 52)], color=PURP, w=2.2, hs=4)
        p.e([(x + 82, y + 70), (x + 38, y + 70)], color=AMBER, w=2.2, hs=4, dashed=True)
        p.t(x + 60, y + 36, "+", fs=14, color=PURP)
        p.t(x + 60, y + 92, "\u2212", fs=16, color=AMBER)

    tiles = [
        (BLUE, Lc("au-au"), ["weighted asymmetric loss", "on AU\u2013AU graph output  (\u00d7 1)", "supervises the intermediate U<sub>a</sub>"],
         g_bars(BLUE, (1, 3))),
        (TEAL, Lc("graph"), ["final graph classifiers", "AU: weighted asymmetric (\u00d7 1)", "expression: BCE vs pseudo-label (\u00d7 1)"], g_two),
        (PURP, Lc("dag"), ["acyclicity of the AU\u2013AU graph", "tr(exp(A \u2299 A)) \u2212 N   (\u00d7 0.1)", "keeps edges directional"], g_dag),
        (AMBER, Lc("facs"), ["FACS rule violations on p", "e.g. p(AU 26) \u00b7 (1 \u2212 p(AU 25))", "(\u00d7 0.05, Phases 2\u20134)"], g_facs),
        (CRIM, Lc("causal"), ["structure violation of edges", "polarity + / \u2212 vs predictions", "(\u00d7 0.05, Phases 3\u20134)"], g_cau),
    ]
    for x, (c, title, lines, g) in zip(xs, tiles):
        tile(p, x, y, tw, th, c, title, lines, g)


def build():
    p = Page("2 \u00b7 Graph layer", W, H, pid="graph_layer")
    p.t(36, 50, "Graph reasoning layer", fs=34, bold=True, color=INK, align="left")
    p.t(36, 88, "Signed global edges \u2299 per-image routing \u00b7 one dense layer shared by the AU\u2013AU and AU\u2192Expression graphs",
        fs=15, color=MUTE, align="left")
    anatomy(p)
    topology(p)
    losses(p)
    return p

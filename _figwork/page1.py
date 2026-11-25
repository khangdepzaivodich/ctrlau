"""Page 1 - end-to-end overview (hero figure)."""
import math
import random
from shapes import *

W, H = 2400, 1250


# ---------------------------------------------------------------- helpers
def rmat(r, c, seed, lo=0.05, hi=1.0, sparse=0.0):
    rnd = random.Random(seed)
    m = [[rnd.uniform(lo, hi) for _ in range(c)] for _ in range(r)]
    for i in range(r):
        for j in range(c):
            if rnd.random() < sparse:
                m[i][j] = lo * 0.4
    return m


def smat(r, c, seed, sparse=0.45):
    rnd = random.Random(seed)
    m = []
    for i in range(r):
        row = []
        for j in range(c):
            v = rnd.uniform(0.3, 1.0) * (1 if rnd.random() > 0.35 else -1)
            if i == j or rnd.random() < sparse:
                v = 0.0
            row.append(v)
        m.append(row)
    return m


def edge_between(a, b, r_a, r_b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    return (a[0] + ux * r_a, a[1] + uy * r_a), (b[0] - ux * r_b, b[1] - uy * r_b)


def minigraph(p, cx, cy, color, seed=1, n=6, rx=34, ry=20, r=6.5, ew=1.6):
    pts = []
    for k in range(n):
        th = math.radians(-90 + 360 * k / n)
        pts.append((cx + rx * math.cos(th), cy + ry * math.sin(th)))
    rnd = random.Random(seed)
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4), (2, 5)]
    for (a, b) in pairs:
        if a >= n or b >= n:
            continue
        if rnd.random() < 0.72:
            s, e = edge_between(pts[a], pts[b], r + 1, r + 1)
            p.e([s, e], color=color, w=ew, hs=3.4, layer=2)
    for (x, y) in pts:
        ell(p, x, y, 2 * r, mix("#FFFFFF", color, 0.28), color, sw=1.3)


def chev(p, x, y, w, h, color, text, fs=14):
    s = st("shape=step", "perimeter=stepPerimeter", "fixedSize=1", "size=16", "html=1", "whiteSpace=nowrap",
           f"fillColor={color}", f"strokeColor={mix('#FFFFFF', color, 0.55)}", "strokeWidth=1.5", "fontColor=#FFFFFF",
           f"fontSize={fs}", "fontStyle=1", f"fontFamily={FONT}", "spacing=0",
           f"gradientColor={mix(color, '#000000', 0.18)}", "gradientDirection=east")
    return p.v(x, y, w, h, s, text, layer=3)


def ticks(p, x, y, w, labels, fs=9.5, color=MUTE):
    n = len(labels)
    bw = w / (n * 1.55)
    gap = (w - n * bw) / (n + 1)
    for i, lab in enumerate(labels):
        p.t(x + gap + i * (bw + gap) + bw / 2, y, lab, fs=fs, color=color, w=26)


AHAT = I("A\u0302")


# ---------------------------------------------------------------- panel A
def branch(p, yb, tag, color, light, tagw=66):
    pill(p, 556 + tagw / 2, yb, tag, color=color, fill=light, fs=12.5, h=26, w=tagw, bold=True, stroke=color)
    x = 646
    cube(p, x, yb - 29, 26, 58, 11, mix("#FFFFFF", color, 0.30), color, grad=mix("#FFFFFF", color, 0.12))
    cube(p, x + 31, yb - 29, 26, 58, 11, mix("#FFFFFF", color, 0.30), color, grad=mix("#FFFFFF", color, 0.12))
    p.e([(x + 62, yb), (x + 84, yb)], color=color, w=1.8)
    trap(p, x + 104, yb, 28, 52, mix("#FFFFFF", color, 0.45), color, narrow="right", inset=13,
         grad=mix("#FFFFFF", color, 0.15))
    p.e([(x + 122, yb), (x + 142, yb)], color=color, w=1.8)
    vec(p, x + 144, yb - 23, 16, 46, 5, color, seed=int(yb))
    p.e([(x + 164, yb), (x + 196, yb)], color=color, w=1.8)


def panelA(p):
    panel(p, 30, 130, 1170, 525, BLUE, "Multi-branch feature learning", num=1, tag="Phase 1")
    cy = 395
    face(p, 62, cy - 50, 100)
    p.t(112, cy + 70, "Input image", fs=13, bold=True)
    p.t(112, cy + 88, "224 \u00d7 224 \u00d7 3", fs=11.5, color=MUTE)
    p.e([(178, cy), (206, cy)], color=LINE, w=2.2)

    trap(p, 262, cy, 120, 196, mix("#FFFFFF", SLATE, 0.78), SLATE, narrow="right", inset=44, grad=SLATE, sw=1.8)
    p.t(258, cy - 9, "ResNet-50", fs=15, bold=True, color="#FFFFFF")
    p.t(258, cy + 10, "SymGraphAU", fs=11, color="#E6EDF5")
    p.t(262, cy + 112, "(B, 49, 2048)", fs=11.5, color=MUTE)
    p.e([(326, cy), (352, cy)], color=LINE, w=2.2)

    trap(p, 372, cy, 34, 86, BLUE_M, BLUE, narrow="right", inset=15, grad=BLUE_L)
    p.t(372, cy + 66, "Linear", fs=11.5, bold=True, color=BLUE)
    p.t(372, cy + 82, "2048\u2192512", fs=11, color=MUTE)
    p.e([(392, cy), (412, cy)], color=LINE, w=2.2)

    tokens(p, 448, cy, n=7, cell=9, gap=1.5, color=BLUE)
    p.t(456, cy + 66, "49 patch tokens", fs=11.5, bold=True, color=BLUE)
    p.t(456, cy + 82, "7 \u00d7 7 grid, 512-d", fs=11, color=MUTE)

    for (y0, name, cnt, color, light, tags, V, rowseed) in [
        (185, "AU branches", "\u00d7 N<sub>AU</sub>", BLUE, BLUE_L, ("AU 1", "AU N"), "a", 7),
        (400, "Expression branches", "\u00d7 8", TEAL, TEAL_L, ("Angry", "Neutral"), "e", 9),
    ]:
        card(p, 530, y0, 420, 205, color, fill="#FFFFFF")
        p.t(550, y0 + 22, name, fs=14, bold=True, color=color, align="left")
        p.t(550 + len(name) * 8.2 + 10, y0 + 22, cnt, fs=12.5, color=MUTE, align="left")
        r1, r2 = y0 + 86, y0 + 166
        branch(p, r1, tags[0], color, light)
        branch(p, r2, tags[1], color, light)
        for dx in (590, 674, 750, 812):
            dots(p, dx, (r1 + r2) / 2, color=color, n=3, gap=7, d=4.0)
        p.t(678, r1 - 41, "Conv \u00d7 2", fs=11, color=MUTE)
        p.t(750, r1 - 41, "Pool", fs=11, color=MUTE)
        p.t(812, r1 - 41, "embedding", fs=11, color=MUTE)
        rows(p, 846, y0 + 30, 80, 160, 8, color, seed=rowseed, pad=6, gap=3)
        p.t(846, y0 + 16, var("V", V), fs=15, color=color, align="left")
        p.t(926, y0 + 17, ("N \u00d7 256" if V == "a" else "8 \u00d7 256"), fs=10.5, color=MUTE, align="right")

    for yb in (185 + 86, 185 + 166, 400 + 86, 400 + 166):
        p.e([(496, cy), (514, cy), (514, yb), (552, yb)], color=LINE, w=2.0, arc=14)

    # auxiliary classifier heads
    for (ya, color, V, lab) in [(310, BLUE, "a", "wa"), (490, TEAL, "e", "we")]:
        p.e([(926, ya), (976, ya)], color=color, w=2.0)
        trap(p, 990, ya, 26, 54, mix("#FFFFFF", color, 0.5), color, narrow="right", inset=10)
        p.t(990, ya + 41, "FC", fs=11, color=MUTE)
        p.e([(1004, ya), (1018, ya)], color=color, w=1.8)
        op(p, 1030, ya, "\u03c3", d=22, fs=13, color=color, stroke=color)
        p.e([(1042, ya), (1056, ya)], color=color, w=1.8)
        barchart(p, 1058, ya - 26, 62, 50, [0.25, 0.7, 0.2, 0.9, 0.35] if V == "a" else [0.1, 0.2, 0.8, 0.15, 0.1],
                 color, hl=(3,) if V == "a" else (2,), hlc=color)
        p.t(1089, ya + 40, var("p", V), fs=13, color=color)
        pill(p, 1166, ya, Lc(lab), color=CRIM, fill=CRIM_L, fs=13, h=26, w=54, bold=True)
        p.e([(1122, ya), (1138, ya)], color=CRIM, w=1.6, dashed=True)

    p.t(1166, 258, var("y", "a"), fs=14, color=INK)
    p.e([(1166, 270), (1166, 296)], color=CRIM, w=1.6, dashed=True)

    heat(p, 1012, 365, 8, prior(), AMBER, lo="#FFF8EC", gap=0.8)
    p.t(1044, 444, var("M", "A\u2013E"), fs=14, color=AMBER)
    p.t(982, 397, var("y", "a"), fs=14, color=INK)
    p.e([(992, 397), (1008, 397)], color=AMBER, w=1.8)
    p.e([(1078, 397), (1098, 397)], color=AMBER, w=1.8)
    pill(p, 1128, 397, "argmax", color=AMBER, fill=AMBER_L, fs=12, h=24, w=58, stroke=AMBER)
    p.e([(1157, 397), (1166, 397), (1166, 476)], color=CRIM, w=1.6, dashed=True, arc=8)
    p.t(1183, 434, var("y", "e"), fs=14, color=INK)

    # phase-1 objective note
    box(p, 62, 160, 440, 98, fill="#FFFFFF", stroke=mix("#FFFFFF", CRIM, 0.5), sw=1.4, arc=10, dashed=True, layer=1)
    p.t(78, 178, "Phase-1 objective", fs=12.5, bold=True, color=CRIM, align="left")
    p.t(78, 200, Lc("wa") + " + 0.05\u00b7" + Lc("we"), fs=13, color=INK, align="left")
    p.t(78, 220, "+ 0.01\u00b7(" + Lc("ib") + " + " + Lc("align") + " + " + Lc("decorr") + ")  <span style='color:#6B7785'>HSIC</span>",
        fs=13, color=INK, align="left")
    p.t(78, 240, "+ 0.05\u00b7(" + Lc("con,AU") + " + " + Lc("con,emo") + ")  <span style='color:#6B7785'>CLIP</span>",
        fs=13, color=INK, align="left")

    # graph exits
    p.e([(926, 244), (1245, 244)], color=BLUE, w=2.6)
    p.t(1100, 228, var("V", "a"), fs=13, color=BLUE)
    p.e([(926, 568), (1235, 568), (1235, 628), (1975, 628), (1975, 590)], color=TEAL, w=2.6, arc=14)
    p.t(1790, 613, var("V", "e") + " bypasses the AU\u2013AU graph", fs=12.5, color=TEAL)

    # CLIP contrastive pill (bottom edge of panel)
    pill(p, 886, 632, Lc("con"), color=AMBER, fill=AMBER_L, fs=13, h=26, w=54, bold=True, stroke=AMBER)
    p.e([(886, 619), (886, 594)], color=AMBER, w=1.8, dashed=True)


# ---------------------------------------------------------------- panel B
def panelB(p):
    panel(p, 1225, 130, 855, 525, PURP, "Causal graph reasoning", num=2, tag="Phases 2\u20134")
    # ---- card 1: AU-AU graph
    x0, y0, w, h = 1245, 185, 380, 405
    card(p, x0, y0, w, h, PURP)
    p.t(x0 + 20, y0 + 24, "AU\u2013AU graph", fs=15, bold=True, color=PURP, align="left")
    p.t(x0 + 20, y0 + 46, "signed global edges \u2299 per-image routing, \u00d7 2 layers", fs=11.5, color=MUTE, align="left")
    labels = ["1", "2", "4", "6", "12", "25", "26", "9"]
    cx, cyy, rx, ry = x0 + 190, y0 + 168, 132, 84
    pts = []
    for k in range(8):
        th = math.radians(-112.5 + 45 * k)
        pts.append((cx + rx * math.cos(th), cyy + ry * math.sin(th)))
    ed = [(0, 1, 1, 3), (1, 2, -1, 2), (3, 4, 1, 3), (4, 5, 1, 2), (5, 6, 1, 3), (7, 0, 1, 1.5), (2, 0, -1, 1.5),
          (7, 2, -1, 1), (3, 1, 1, 1.2), (4, 6, 1, 1.4), (6, 7, 1, 1), (5, 3, 1, 1.2), (2, 3, 1, 1.1), (7, 6, -1, 1.2)]
    for (a, b, sg, wt) in ed:
        s, e = edge_between(pts[a], pts[b], 17, 17)
        p.e([s, e], color=PURP if sg > 0 else AMBER, w=0.9 + wt * 0.55, hs=3.2, opacity=100 if wt > 1.4 else 70,
            dashed=(sg < 0))
    for k, (x, y) in enumerate(pts):
        node(p, x, y, labels[k], "au", d=32)

    ry = y0 + 336
    heat(p, x0 + 24, ry - 30, 10, smat(6, 6, 3), PURP, lo="#FAF8FC", signed=(PURP, AMBER), gap=0.8)
    op(p, x0 + 106, ry, "\u2299", d=24, fs=15)
    heat(p, x0 + 126, ry - 30, 10, rmat(6, 6, 5, lo=0.1, hi=0.95), PURP, lo="#FAF8FC", gap=0.8)
    p.e([(x0 + 196, ry), (x0 + 232, ry)], color=PURP, w=1.8)
    p.t(x0 + 214, ry - 14, "norm.", fs=10.5, color=MUTE)
    heat(p, x0 + 242, ry - 30, 10, [[abs(v) for v in r] for r in smat(6, 6, 8, 0.35)], PURP, lo="#FAF8FC", gap=0.8)
    p.t(x0 + 54, ry + 44, "global  " + var("A"), fs=11.5, color=MUTE)
    p.t(x0 + 156, ry + 44, "routing  " + var("R"), fs=11.5, color=MUTE)
    p.t(x0 + 272, ry + 44, AHAT + "  used", fs=11.5, color=MUTE)

    # U_a between the cards
    ux = 1654
    p.e([(x0 + w, 380), (ux - 2, 380)], color=PURP, w=2.4)
    rows(p, ux, 305, 44, 150, 8, PURP, seed=21, pad=5, gap=3)
    p.t(ux + 22, 286, var("U", "a"), fs=15, color=PURP)
    p.t(ux + 22, 472, "updated AUs", fs=11, color=MUTE)
    p.e([(ux + 46, 380), (1723, 380)], color=PURP, w=2.4)

    # ---- card 2: AU -> Expression
    X0, Y0, Wd, Hd = 1725, 185, 330, 405
    card(p, X0, Y0, Wd, Hd, PURP)
    p.t(X0 + 20, Y0 + 24, "AU \u2192 Expression graph", fs=15, bold=True, color=PURP, align="left")
    p.t(X0 + 20, Y0 + 46, "one-way: expressions never message AUs", fs=11.5, color=MUTE, align="left")
    ax, ex = X0 + 62, X0 + 262
    ys = [Y0 + 114 + 42 * i for i in range(5)]
    au_l = ["1", "2", "6", "12", "25"]
    ex_l = ["Ang", "Fea", "Hap", "Sad", "Sur"]
    p.t(ax, Y0 + 82, "AU nodes", fs=11.5, color=MUTE)
    p.t(ex, Y0 + 82, "Expression nodes", fs=11.5, color=MUTE)
    strong = {(0, 4): 2.6, (1, 4): 2.6, (4, 4): 1.8, (2, 2): 2.8, (3, 2): 2.8, (0, 1): 1.8, (1, 1): 1.8, (3, 0): 1.2}
    for i in range(5):
        for j in range(5):
            s, e = edge_between((ax, ys[i]), (ex, ys[j]), 16, 16)
            if (i, j) in strong:
                p.e([s, e], color=PURP, w=strong[(i, j)], hs=3.6, layer=2)
            else:
                p.e([s, e], color=PURP_M, w=0.8, head=None, opacity=45, layer=2)
    for i in range(5):
        node(p, ax, ys[i], au_l[i], "au", d=32)
        node(p, ex, ys[i], ex_l[i], "ex", d=34, fs=10.5)
    dots(p, ax, ys[-1] + 32, color=TERRA, n=3, gap=7, d=4.0)
    dots(p, ex, ys[-1] + 32, color="#4F73B5", n=3, gap=7, d=4.0)

    bx, by, bs = X0 + 28, Y0 + 334, 28
    for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        learn = (i, j) == (0, 1)
        sq(p, bx + j * bs, by + i * bs, bs, bs, PURP_M if learn else "#F4F2F7", stroke="#FFFFFF", sw=1.2)
    p.v(bx, by, 2 * bs, 2 * bs, st("rounded=0", "html=1", "fillColor=none", f"strokeColor={PURP}", "strokeWidth=1.4"))
    p.t(bx + bs * 1.5, by + bs * 0.5, var("A", "AE"), fs=10.5, color=PURP)
    for (i, j) in [(0, 0), (1, 0), (1, 1)]:
        p.t(bx + bs * (j + 0.5), by + bs * (i + 0.5), "0", fs=11, color=FAINT)
    p.t(bx + 2 * bs + 14, by + 14, "only the AU\u2192Expr block", fs=11.5, color=MUTE, align="left")
    p.t(bx + 2 * bs + 14, by + 31, "is learnable", fs=11.5, color=MUTE, align="left")
    p.t(bx + 2 * bs + 14, by + 48, "(" + var("A", "AE") + " : N \u00d7 8)", fs=11.5, color=MUTE, align="left")

    # outputs fork to the two classifiers
    p.e([(X0 + Wd, 380), (2092, 380), (2092, 270), (2126, 270)], color=PURP, w=2.4, arc=12)
    p.e([(2092, 380), (2092, 497), (2126, 497)], color=PURP, w=2.4, arc=12)


# ---------------------------------------------------------------- panel C
def panelC(p):
    panel(p, 2105, 130, 270, 525, GREEN, "Prediction")
    for (cy_, color, title, labels, vals, hl, lab) in [
        (270, BLUE, "AU probabilities", ["1", "2", "4", "6", "9", "12", "25", "26"],
         [0.12, 0.18, 0.1, 0.92, 0.15, 0.88, 0.3, 0.08], (3, 5), "graph-au"),
        (497, TEAL, "Expression probabilities", ["Ang", "Fea", "Hap", "Sad", "Sur", "Dis", "Con", "Neu"],
         [0.04, 0.06, 0.86, 0.05, 0.1, 0.03, 0.05, 0.02], (2,), "graph-emo"),
    ]:
        p.t(2240, cy_ - 64, title, fs=13.5, bold=True, color=color)
        p.e([(2126, cy_), (2134, cy_)], color=color, w=1.8, head=None)
        trap(p, 2146, cy_, 24, 52, mix("#FFFFFF", color, 0.5), color, narrow="right", inset=10)
        p.t(2146, cy_ + 40, "FC", fs=11, color=MUTE)
        p.e([(2158, cy_), (2169, cy_)], color=color, w=1.8)
        op(p, 2181, cy_, "\u03c3", d=22, fs=13, color=color, stroke=color)
        p.e([(2193, cy_), (2204, cy_)], color=color, w=1.8)
        barchart(p, 2206, cy_ - 36, 150, 72, vals, color, hl=hl, hlc=color)
        ticks(p, 2206, cy_ + 48, 150, labels, fs=9.5 if len(labels[0]) < 3 else 8.5)
        pill(p, 2240, cy_ + 84, Lc(lab), color=CRIM, fill=CRIM_L, fs=12.5, h=24, w=96, bold=True)
        p.e([(2240, cy_ + 62), (2240, cy_ + 71)], color=CRIM, w=1.6, dashed=True, head="none")
    pill(p, 2240, 628, "AU 6 + AU 12  \u21d2  Happy", color="#2E6B31", fill=GREEN_L, fs=13, h=30, w=212, bold=True,
         stroke=GREEN)


# ---------------------------------------------------------------- tier 2
def curriculum_panel(p):
    panel(p, 30, 735, 560, 395, GREY, "Training curriculum", tag="warm-start", chip_x=26)
    phases = [
        (BLUE, "P1", "Feature learning", "trains backbone, branches, projections",
         Lc("wa") + " \u00b7 " + Lc("we") + " \u00b7 HSIC \u00b7 " + Lc("con")),
        (PURP, "P2", "Graph learning", "trains both graphs + classifiers; features frozen",
         Lc("au-au") + " \u00b7 " + Lc("graph") + " \u00b7 " + Lc("dag") + " \u00b7 " + Lc("facs")),
        (CRIM, "P3", "Counterfactual", "adds masks and cutoffs \u03c4; visual path frozen",
         "+ " + Lc("causal") + " \u00b7 " + Lc("cf") + " (4 terms)"),
        (GREEN, "P4", "End-to-end", "also fine-tunes the visual path",
         "same objective as P3"),
    ]
    for k, (c, tag, name, a, b) in enumerate(phases):
        y = 800 + k * 76
        chev(p, 58, y, 92, 46, c, tag)
        p.t(168, y + 10, name, fs=14, bold=True, color=c if c != GREEN else "#2E6B31", align="left")
        p.t(168, y + 29, a, fs=11.5, color=MUTE, align="left")
        p.t(168, y + 46, b, fs=12, color=INK, align="left")
        if k < 3:
            p.e([(104, y + 50), (104, y + 72)], color=GREY, w=2.0, hs=4)
    p.t(310, 1108, "each phase warm-starts from the previous checkpoint", fs=11.5, color=MUTE, italic=True)


def clip_panel(p):
    panel(p, 620, 735, 580, 395, AMBER, "Semantic prior from CLIP", num=3, tag="frozen")
    cyc = 925
    nx, ny = 646, cyc - 65
    p.v(nx, ny, 170, 130, st("shape=note", "size=16", "whiteSpace=wrap", "html=1", "backgroundOutline=1",
                             "fillColor=#FFFFFF", f"strokeColor={AMBER}", "strokeWidth=1.5"), layer=3)
    lines = ["AU 1  Inner Brow Raiser", "AU 2  Outer Brow Raiser", "AU 6  Cheek Raiser", "\u22ee",
             "Happy: primary AU 6, 12"]
    for i, ln in enumerate(lines):
        p.t(nx + 12, ny + 26 + i * 20, ln, fs=10.5, color=INK if i != 3 else MUTE, align="left")
    p.t(nx + 85, ny - 22, "AU & expression text", fs=12.5, bold=True, color=AMBER)
    p.e([(820, cyc), (850, cyc)], color=AMBER, w=2.2)

    trap(p, 910, cyc, 110, 170, mix("#FFFFFF", AMBER, 0.75), AMBER, narrow="right", inset=48, grad=AMBER, sw=1.8)
    p.t(906, cyc - 9, "CLIP", fs=17, bold=True, color="#FFFFFF")
    p.t(906, cyc + 12, "text encoder", fs=11.5, color="#FFF4DC")
    frozen(p, 868, cyc - 72)
    p.t(910, cyc + 100, "ViT-B/32 text tower", fs=11, color=MUTE)
    p.e([(968, cyc), (1008, cyc)], color=AMBER, w=2.2)

    rows(p, 1012, cyc - 58, 76, 116, 6, AMBER, seed=11, pad=6, gap=4)
    p.t(1050, cyc + 76, "text anchors " + var("t"), fs=12.5, bold=True, color=AMBER)
    p.t(1050, cyc + 94, "512-d \u2192 shared space", fs=11, color=MUTE)
    # to L_con in panel 1 (elbow through the gap between the tiers)
    p.e([(1050, cyc - 62), (1050, 696), (886, 696), (886, 646)], color=AMBER, w=2.2, arc=14)
    p.t(968, 682, "aligns embeddings (Phase 1)", fs=11.5, color=AMBER)
    # to the counterfactual panel
    p.e([(1092, cyc), (1212, cyc), (1212, 1106), (1657, 1106), (1657, 1081)], color=AMBER, w=2.2, arc=14)
    p.t(1430, 1106, "semantic directions " + var("u") + " from the text anchors", fs=11.5, color=AMBER,
        bg=mix("#FFFFFF", CRIM, 0.06), w=250)


def cf_panel(p):
    panel(p, 1225, 735, 1150, 395, CRIM, "Counterfactual", num=4, tag="training only")
    p.t(1255, 770, "intervention on the factual routing \u00b7 Phases 3\u20134 \u00b7 applied at two levels: AU\u2192AU and AU\u2192Expression",
        fs=12, color=MUTE, italic=True, align="left")
    X = [1245 + 286 * k for k in range(4)]
    ytop, hh, cw_ = 800, 280, 252
    for x in X:
        card(p, x, ytop, cw_, hh, CRIM, fill="#FFFFFF", arc=12)
    for k in range(3):
        p.e([(X[k] + cw_ + 2, ytop + 135), (X[k + 1] - 2, ytop + 135)], color=CRIM, w=2.4)

    # --- 1: soft masks
    x = X[0]
    badge(p, x + 22, ytop + 24, "a", CRIM, d=24)
    p.t(x + 40, ytop + 24, "Soft importance masks", fs=13.5, bold=True, color=CRIM, align="left")
    p.t(x + 126, ytop + 54, "M = \u03c3(\u03b3\u00b7(s \u2212 \u03c4)),  \u03b3 = 5", fs=11.5, color=CRIM, family=SERIF)
    r = rmat(7, 7, 12, lo=0.05, hi=1.0, sparse=0.3)
    heat(p, x + 16, ytop + 92, 11, r, PURP, lo="#FAF8FC", gap=0.8)
    p.t(x + 54, ytop + 186, var("s") + " = routing / mean", fs=11, color=MUTE)
    p.e([(x + 98, ytop + 130), (x + 118, ytop + 130)], color=CRIM, w=1.8)
    mi = [[1 if v > 0.55 else 0.1 for v in row] for row in r]
    mu = [[1.1 - v for v in row] for row in mi]
    heat(p, x + 124, ytop + 82, 7, mi, CRIM, lo="#FDF3F5", gap=0.7)
    heat(p, x + 124, ytop + 150, 7, mu, TEAL, lo="#F1FAF8", gap=0.7)
    p.t(x + 182, ytop + 107, var("M", "imp"), fs=11.5, color=CRIM, align="left")
    p.t(x + 182, ytop + 175, var("M", "unimp"), fs=11.5, color=TEAL, align="left")
    p.t(x + 126, ytop + 224, "important = above-average", fs=11, color=MUTE)
    p.t(x + 126, ytop + 240, "incoming routing", fs=11, color=MUTE)

    # --- 2: semantic do() intervention
    x = X[1]
    badge(p, x + 22, ytop + 24, "b", CRIM, d=24)
    p.t(x + 40, ytop + 24, "Semantic do(\u00b7) edit", fs=13.5, bold=True, color=CRIM, align="left")
    hvec(p, x + 36, ytop + 62, 180, 22, 8, BLUE, seed=2)
    p.t(x + 20, ytop + 73, var("h"), fs=14, color=BLUE)
    hvec(p, x + 20, ytop + 134, 98, 22, 8, CRIM, active={0, 3, 4, 6, 7})
    hvec(p, x + 134, ytop + 134, 98, 22, 8, TEAL, active={1, 2, 5})
    p.e([(x + 69, ytop + 87), (x + 69, ytop + 130)], color=CRIM, w=1.8)
    p.e([(x + 183, ytop + 87), (x + 183, ytop + 130)], color=TEAL, w=1.8)
    p.t(x + 69, ytop + 176, "h \u2212 \u03b1c\u2299" + var("M", "imp"), fs=11, color=CRIM)
    p.t(x + 183, ytop + 176, "h \u2212 \u03b1c\u2299" + var("M", "unimp"), fs=11, color=TEAL)
    p.t(x + 126, ytop + 214, "c = \u27e8h, u\u27e9 u   (CLIP direction u)", fs=11, color=AMBER)
    p.t(x + 126, ytop + 232, "hollow cell = removed component", fs=11, color=MUTE)
    p.t(x + 126, ytop + 250, "\u03b1 \u2264 1, no overshoot", fs=11, color=MUTE)

    # --- 3: re-run graph
    x = X[2]
    badge(p, x + 22, ytop + 24, "c", CRIM, d=24)
    p.t(x + 40, ytop + 24, "Re-run the graph", fs=13.5, bold=True, color=CRIM, align="left")
    minigraph(p, x + 138, ytop + 100, CRIM, seed=3, n=6, rx=52, ry=26, r=8)
    minigraph(p, x + 138, ytop + 188, TEAL, seed=5, n=6, rx=52, ry=26, r=8)
    p.t(x + 22, ytop + 100, "imp", fs=11, color=CRIM, align="left", italic=True)
    p.t(x + 22, ytop + 188, "unimp", fs=11, color=TEAL, align="left", italic=True)
    p.t(x + 126, ytop + 240, "same weights, perturbed input", fs=11, color=MUTE)
    p.t(x + 126, ytop + 257, "read out the target node", fs=11, color=MUTE)

    # --- 4: compare
    x = X[3]
    badge(p, x + 22, ytop + 24, "d", CRIM, d=24)
    p.t(x + 40, ytop + 24, "Compare outputs", fs=13.5, bold=True, color=CRIM, align="left")
    bw_ = 62
    for k, (nm, vals, c) in enumerate([("\u0177", [0.2, 0.8, 0.3, 0.7, 0.2], GREY),
                                       ("\u0177\u2032", [0.6, 0.3, 0.7, 0.2, 0.55], CRIM),
                                       ("\u0177\u2033", [0.22, 0.78, 0.32, 0.68, 0.2], TEAL)]):
        bx = x + 16 + k * (bw_ + 14)
        barchart(p, bx, ytop + 74, bw_, 66, vals, c, hl=range(5), hlc=c)
        p.t(bx + bw_ / 2, ytop + 158, nm, fs=14, color=c, family=SERIF)
    p.t(x + 16 + (bw_ + 14) * 1 + bw_ / 2, ytop + 180, "must change", fs=10.5, color=CRIM)
    p.t(x + 16 + (bw_ + 14) * 2 + bw_ / 2, ytop + 180, "must stay", fs=10.5, color=TEAL)
    pill(p, x + 126, ytop + 222, Lc("cf-imp") + "  push apart", color=CRIM, fill=CRIM_L, fs=12, h=24, w=176)
    pill(p, x + 126, ytop + 252, Lc("cf-unimp") + "  keep close", color="#1F6E63", fill=TEAL_L, fs=12, h=24, w=176,
         stroke=TEAL)

    # factual routing from the AU-AU graph into step 1
    p.e([(1470, 590), (1470, ytop)], color=PURP, w=2.0, dashed=True, extra="jumpStyle=arc;jumpSize=10")
    p.t(1478, 692, "factual routing " + AHAT + " from the graph", fs=11.5, color=PURP, align="left", bg="#FFFFFF")


def legend(p):
    y = 1190
    p.t(36, y, "Legend", fs=13, bold=True, align="left")
    x = 118

    def item(kind, label, iw):
        nonlocal x
        cx = x + iw / 2
        kind(cx)
        tx = x + iw + 8
        p.t(tx, y, label, fs=12, color=MUTE, align="left")
        n, _ = plain_len(label)
        x = tx + n * 6.6 + 30

    item(lambda cx: node(p, cx, y, "1", "au", d=24, fs=10), "AU node", 24)
    item(lambda cx: node(p, cx, y, "Hap", "ex", d=28, fs=8.5), "expression node", 28)
    item(lambda cx: vec(p, cx - 7, y - 12, 14, 24, 4, BLUE, seed=1), "embedding", 14)
    item(lambda cx: heat(p, cx - 12, y - 12, 8, [[0.2, 0.9, 0.4], [0.7, 0.1, 0.6], [0.3, 0.5, 0.95]], PURP,
                         lo="#FAF8FC", gap=0.6), "adjacency / mask", 24)
    item(lambda cx: barchart(p, cx - 14, y - 12, 28, 24, [0.4, 0.9, 0.3], BLUE, hl=(1,), hlc=BLUE), "probability", 28)
    item(lambda cx: p.e([(cx - 18, y), (cx + 18, y)], color=LINE, w=2.2), "data flow", 36)
    item(lambda cx: p.e([(cx - 18, y), (cx + 18, y)], color=PURP, w=2.2), "graph message", 36)
    item(lambda cx: p.e([(cx - 18, y), (cx + 18, y)], color=AMBER, w=2.2), "text / prior", 36)
    item(lambda cx: p.e([(cx - 18, y), (cx + 18, y)], color=CRIM, w=1.8, dashed=True), "supervision / loss", 36)
    item(lambda cx: p.e([(cx - 18, y), (cx + 18, y)], color=AMBER, w=1.8, dashed=True, head="none"), "inhibitory edge", 36)
    item(lambda cx: frozen(p, cx, y, d=20), "frozen", 20)
    p.t(2364, y, "DISFA AUs 1, 2, 4, 6, 9, 12, 25, 26  \u00b7  BP4D AUs 1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24",
        fs=11.5, color=FAINT, align="right")


def build():
    p = Page("1 \u00b7 Overview", W, H, pid="overview")
    p.t(36, 50, "CtrlAU", fs=34, bold=True, color=INK, align="left")
    p.t(36, 88, "Causal AU graph reasoning with CLIP-guided counterfactual regularization \u00b7 end-to-end architecture",
        fs=15, color=MUTE, align="left")
    pill(p, 2364 - 292, 62, "<b>Inference path:</b>  1 \u2192 2 \u2192 Prediction   \u00b7   3, 4 are training-only",
         color=INK, fill="#F3F5F8", fs=13, h=34, w=584, stroke=GREY_M)
    panelA(p)
    panelB(p)
    panelC(p)
    curriculum_panel(p)
    clip_panel(p)
    cf_panel(p)
    legend(p)
    return p

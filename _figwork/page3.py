"""Page 3 - counterfactual intervention in detail."""
import math
import random
from shapes import *
from page1 import rmat, smat, edge_between, minigraph, ticks, AHAT, chev

W, H = 2400, 1140
CARD_W, GAP, X0, Y0, CARD_H = 350, 38, 60, 175, 560
SERIF_T = dict(family=SERIF)


def stack3d(p, x, y, w, h, color, n=3, seed=0, dx=9, dy=9):
    for k in range(n - 1, -1, -1):
        rows(p, x + k * dx, y - k * dy, w, h, 5, color, seed=seed + k, pad=4, gap=2, lo=0.25 if k == 0 else 0.1)


def head(p, k, title, sub):
    x = X0 + k * (CARD_W + GAP)
    badge(p, x + 24, Y0 + 28, str(k + 1), CRIM, d=26)
    p.t(x + 46, Y0 + 28, title, fs=15.5, bold=True, color=CRIM, align="left")
    p.t(x + 20, Y0 + 56, sub, fs=11.5, color=MUTE, align="left", italic=True)
    return x


def s1(p):
    x = head(p, 0, "Factual pass", "run the graph as usual")
    J = (x + 175, Y0 + 212)
    srcs = [(x + 58, Y0 + 146, "i\u2081", 1.4), (x + 128, Y0 + 104, "i\u2082", 3.6),
            (x + 222, Y0 + 104, "i\u2083", 1.6), (x + 292, Y0 + 146, "i\u2084", 3.0)]
    for (sx, sy, lab, wt) in srcs:
        s, e = edge_between((sx, sy), J, 17, 19)
        p.e([s, e], color=PURP, w=wt, hs=3.6)
    for (sx, sy, lab, wt) in srcs:
        node(p, sx, sy, lab, "au", d=34, fs=12)
    ell(p, J[0], J[1], 40, CRIM_L, CRIM, sw=2.4, value="j", fs=14, bold=True, color=CRIM)
    p.t(x + 175, Y0 + 262, "edge width = routing weight into target j", fs=11.5, color=MUTE)
    c = 17
    hx, hy = x + 124, Y0 + 304
    heat(p, hx, hy, c, rmat(6, 6, 14, lo=0.05, hi=1.0, sparse=0.3), PURP, lo="#FAF8FC", gap=0.9)
    p.v(hx - 2, hy + 3 * c - 2, 6 * c + 4, c + 4,
        st("rounded=0", "html=1", "fillColor=none", f"strokeColor={CRIM}", "strokeWidth=2.4"), layer=4)
    p.t(hx + 3 * c, hy - 12, "source i", fs=11, color=MUTE)
    p.t(hx - 18, hy + 3 * c, "target", fs=11, color=MUTE, rot=270, w=50)
    p.t(hx + 6 * c + 12, hy + 3 * c + c / 2, "row j", fs=11.5, color=CRIM, align="left", bold=True)
    p.t(hx + 3 * c, hy + 6 * c + 16, "|" + AHAT + "|  routing weights", fs=13, color=PURP)
    p.t(x + 20, Y0 + 452, AHAT + " is the signed routing actually used in the", fs=12.5, color=INK, align="left")
    p.t(x + 20, Y0 + 473, "forward pass, averaged over the 2 layers.", fs=12.5, color=INK, align="left")
    p.t(x + 20, Y0 + 503, "Outputs Z and p are detached: they are the", fs=12.5, color=MUTE, align="left")
    p.t(x + 20, Y0 + 524, "reference for both perturbed passes.", fs=12.5, color=MUTE, align="left")


def s2(p):
    x = head(p, 1, "Soft importance mask", "important = above-average incoming routing")
    c = 15
    hx, hy = x + 24, Y0 + 100
    r = rmat(6, 6, 14, lo=0.05, hi=1.0, sparse=0.3)
    heat(p, hx, hy, c, r, PURP, lo="#FAF8FC", gap=0.9)
    p.t(hx + 3 * c, hy + 6 * c + 16, var("s") + " = |" + AHAT + "| / mean<sub>i</sub> |" + AHAT + "|", fs=12.5, color=PURP)
    bx, by = hx + 4, hy + 6 * c + 40
    for k in range(20):
        sq(p, bx + k * 4.3, by, 4.3, 9, mix("#FAF8FC", PURP, k / 19), stroke="none")
    p.e([(bx + 43, by - 3), (bx + 43, by + 13)], color=CRIM, w=2.0, head=None, rounded=False, layer=4)
    p.t(bx, by + 24, "0", fs=10.5, color=MUTE)
    p.t(bx + 86, by + 24, "2+", fs=10.5, color=MUTE)
    p.t(bx + 43, by + 24, "\u03c4 = 1", fs=10.5, color=CRIM)
    ym = hy + 3 * c
    p.e([(hx + 6 * c + 8, ym), (hx + 6 * c + 36, ym)], color=CRIM, w=2.0)
    op(p, hx + 6 * c + 54, ym, "\u03c3", d=32, fs=13, color=CRIM, stroke=CRIM)
    mx = x + 236
    ya_, yb_ = hy + 30, hy + 116
    p.e([(hx + 6 * c + 70, ym), (x + 196, ym), (x + 196, ya_), (mx - 6, ya_)], color=CRIM, w=2.0, arc=8)
    p.e([(x + 196, ym), (x + 196, yb_), (mx - 6, yb_)], color=TEAL, w=2.0, arc=8)
    p.t(x + 206, yb_ - 14, "1 \u2212", fs=11.5, color=TEAL, family=SERIF, italic=True, w=30)
    mi = [[1 if v > 0.55 else 0.08 for v in row] for row in r]
    mu = [[1.08 - v for v in row] for row in mi]
    heat(p, mx, ya_ - 33, 11, mi, CRIM, lo="#FDF3F5", gap=0.8)
    heat(p, mx, yb_ - 33, 11, mu, TEAL, lo="#F1FAF8", gap=0.8)
    p.t(mx + 33, ya_ - 45, var("M", "imp"), fs=13, color=CRIM)
    p.t(mx + 33, yb_ + 45, var("M", "unimp"), fs=13, color=TEAL)
    p.t(x + 20, Y0 + 372, "M = \u03c3( \u03b3 \u00b7 ( s \u2212 \u03c4 ) ),   \u03b3 = 5", fs=15, color=CRIM, family=SERIF, align="left")
    p.t(x + 20, Y0 + 408, "\u03c4 = softplus(\u03c4\u0303): learnable, starts at 1", fs=12.5, color=INK, align="left")
    p.t(x + 20, Y0 + 430, "(each target's mean incoming routing)", fs=12.5, color=MUTE, align="left")
    p.t(x + 20, Y0 + 462, "Absent edges get zero importance.", fs=12.5, color=INK, align="left")
    p.t(x + 20, Y0 + 484, "A heuristic, not an identified causal claim.", fs=12.5, color=MUTE, align="left")


def s3(p):
    x = head(p, 2, "Semantic component", "CLIP gives the direction to remove")
    ox, oy = x + 70, Y0 + 262
    ux, uy = math.cos(math.radians(24)), -math.sin(math.radians(24))
    hv = (186, -120)
    dotp = hv[0] * ux + hv[1] * uy
    cvec = (dotp * ux, dotp * uy)
    p.e([(ox - 14, oy), (ox + 250, oy)], color=GREY_M, w=1.4, head=None, rounded=False)
    p.e([(ox, oy + 14), (ox, oy - 170)], color=GREY_M, w=1.4, head=None, rounded=False)
    p.e([(ox, oy), (ox + hv[0], oy + hv[1])], color=BLUE, w=2.8, hs=5)
    p.e([(ox, oy), (ox + cvec[0], oy + cvec[1])], color=AMBER, w=2.8, hs=5)
    p.e([(ox + cvec[0], oy + cvec[1]), (ox + hv[0], oy + hv[1])], color=GREY, w=1.6, dashed=True, head="none")
    p.e([(ox, oy), (ox + ux * 62, oy + uy * 62)], color="#8A5A00", w=3.4, hs=6)
    p.t(ox + hv[0] + 6, oy + hv[1] - 14, var("h", "i"), fs=17, color=BLUE)
    p.t(ox + cvec[0] + 18, oy + cvec[1] + 14, var("c", "i"), fs=17, color=AMBER)
    p.t(ox + ux * 62 + 14, oy + uy * 62 + 20, var("u", "i"), fs=16, color="#8A5A00")
    p.t(ox + hv[0] + 34, oy + (hv[1] + cvec[1]) / 2 + 2, "h \u2212 c", fs=12, color=GREY, family=SERIF, italic=True, w=50)
    p.t(x + 175, Y0 + 86, "feature space of source AU i", fs=11.5, color=MUTE)
    cy_ = Y0 + 336
    pill(p, x + 56, cy_, "text " + var("t", "i"), color=AMBER, fill=AMBER_L, fs=12.5, h=28, w=82, stroke=AMBER)
    p.e([(x + 98, cy_), (x + 122, cy_)], color=AMBER, w=1.8)
    pill(p, x + 190, cy_, "text_proj \u2192 " + var("g", "i"), color=AMBER, fill=AMBER_L, fs=12.5, h=28, w=122, stroke=AMBER)
    p.e([(x + 252, cy_), (x + 280, cy_)], color=AMBER, w=1.8)
    frozen(p, x + 300, cy_, d=26)
    p.t(x + 20, Y0 + 386, var("u", "i") + " = \u2207<sub>h</sub> \u27e8 visual_proj(" + var("h", "i") + "),  " + var("g", "i") + " \u27e9 ,   \u2016u\u2016 = 1",
        fs=12.5, color=INK, align="left")
    p.t(x + 20, Y0 + 418, var("c", "i") + " = \u27e8 " + var("h", "i") + ", " + var("u", "i") + " \u27e9 " + var("u", "i"),
        fs=15, color=AMBER, align="left")
    p.t(x + 20, Y0 + 456, "u is a local, detached direction (gradient of", fs=12.5, color=MUTE, align="left")
    p.t(x + 20, Y0 + 477, "the CLIP score), not an inverse of visual_proj.", fs=12.5, color=MUTE, align="left")
    nb = "\u00a0"
    p.t(x + 20, Y0 + 512, "Level 1:" + nb * 2 + "h = V<sub>a</sub>" + nb * 8 + "Level 2:" + nb * 2 + "h = U<sub>a</sub>", fs=12.5,
        color=INK, align="left")


def s4(p):
    x = head(p, 3, "do(\u00b7) intervention", "remove the semantic part, per edge")
    stack3d(p, x + 52, Y0 + 122, 150, 80, CRIM, n=3, seed=3)
    p.t(x + 52 + 75 + 9, Y0 + 86, "important removed", fs=12.5, color=CRIM, bold=True)
    stack3d(p, x + 52, Y0 + 288, 150, 80, TEAL, n=3, seed=9)
    p.t(x + 52 + 75 + 9, Y0 + 252, "unimportant removed", fs=12.5, color=TEAL, bold=True)
    for yy in (Y0 + 168, Y0 + 334):
        p.t(x + 290, yy, "\u00d7 N", fs=14, color=MUTE, family=SERIF, italic=True)
        p.t(x + 290, yy + 18, "targets j", fs=11, color=MUTE)
    p.t(x + 20, Y0 + 410, "h\u0303<sub>j,i</sub> = h<sub>i</sub> \u2212 \u03b1 \u00b7 c<sub>i</sub> \u00b7 M<sub>ji</sub>", fs=16, color=CRIM,
        family=SERIF, align="left")
    p.t(x + 20, Y0 + 446, "M = M<sub>imp</sub> (crimson)  or  1 \u2212 M<sub>imp</sub> (teal)", fs=12.5, color=INK, align="left")
    p.t(x + 20, Y0 + 470, "One perturbed copy of all sources per target j.", fs=12.5, color=MUTE, align="left")
    p.t(x + 20, Y0 + 492, "\u03b1 \u2264 1: bounded attenuation, no overshoot.", fs=12.5, color=MUTE, align="left")
    p.t(x + 20, Y0 + 514, "Gradients flow through h and the soft masks.", fs=12.5, color=MUTE, align="left")


def s5(p):
    x = head(p, 4, "Re-run the graph", "same weights, perturbed inputs")
    for (yy, c, lab, sd) in [(Y0 + 138, CRIM, "important removed", 3), (Y0 + 292, TEAL, "unimportant removed", 5)]:
        minigraph(p, x + 130, yy, c, seed=sd, n=6, rx=84, ry=40, r=13)
        p.e([(x + 232, yy), (x + 262, yy)], color=c, w=2.0)
        trap(p, x + 276, yy, 24, 50, mix("#FFFFFF", c, 0.45), c, narrow="right", inset=9)
        p.e([(x + 290, yy), (x + 312, yy)], color=c, w=2.0)
        p.t(x + 326, yy, "\u0177<sub>j</sub>", fs=15, color=c, family=SERIF, italic=True, w=40)
        p.t(x + 130, yy + 62, lab, fs=12, color=c, bold=True)
        p.t(x + 276, yy + 38, "clf<sub>j</sub>", fs=10.5, color=MUTE)
    p.t(x + 20, Y0 + 410, "Read out node j of copy j only:", fs=12.5, color=INK, align="left")
    p.t(x + 20, Y0 + 436, "feature  Z\u0303<sub>j</sub>   and probability  p\u0303<sub>j</sub>", fs=14, color=INK, align="left", family=SERIF)
    p.t(x + 20, Y0 + 472, "Graph weights and classifiers are the same", fs=12.5, color=MUTE, align="left")
    p.t(x + 20, Y0 + 493, "tensors as in the factual pass (shared).", fs=12.5, color=MUTE, align="left")


def angle_glyph(p, cx, cy, deg, color, label, sub_, zdy=0):
    r = 62
    a0 = math.radians(-24)
    a1 = math.radians(-24 - deg)
    ox, oy = cx - 6, cy + 36
    p.e([(ox, oy), (ox + r * math.cos(a0), oy + r * math.sin(a0))], color=GREY, w=2.6, hs=4.5)
    r1 = r * (0.9 if deg < 20 else 1.0)
    p.e([(ox, oy), (ox + r1 * math.cos(a1), oy + r1 * math.sin(a1))], color=color, w=2.6, hs=4.5)
    zx, zy = ox + (r + 16) * math.cos(a0), oy + (r + 16) * math.sin(a0) + zdy
    p.t(zx, zy, "Z", fs=14, color=GREY, family=SERIF, italic=True, w=20)
    lx, ly = ox + (r1 + 17) * math.cos(a1), oy + (r1 + 17) * math.sin(a1)
    p.t(lx, ly - (8 if zdy else 0), label, fs=14, color=color, family=SERIF, italic=True, w=34)
    p.t(cx, cy + 84, sub_, fs=12, color=color, bold=True)


def s6(p):
    x = head(p, 5, "Counterfactual losses", "compare with the factual reference")
    angle_glyph(p, x + 100, Y0 + 110, 150, CRIM, "Z\u0303\u2032", "diverge: cos \u2192 \u22121")
    angle_glyph(p, x + 262, Y0 + 110, 10, TEAL, "Z\u0303\u2033", "agree: cos \u2192 1", zdy=14)
    p.t(x + 20, Y0 + 262, Lc("cf-imp"), fs=16, color=CRIM, align="left", bold=True)
    p.t(x + 20, Y0 + 290, "= \u03b7 ( 1 + cos(Z, Z\u0303\u2032) )", fs=14, color=INK, align="left", family=SERIF)
    p.t(x + 20, Y0 + 314, "+ \u03b7 max( 0,  1 \u2212 N \u00b7 MSE(p, p\u0303\u2032) )", fs=14, color=INK, align="left", family=SERIF)
    p.t(x + 20, Y0 + 354, Lc("cf-unimp"), fs=16, color=TEAL, align="left", bold=True)
    p.t(x + 20, Y0 + 382, "= \u03b4 ( 1 \u2212 cos(Z, Z\u0303\u2033) )", fs=14, color=INK, align="left", family=SERIF)
    p.t(x + 20, Y0 + 406, "+ \u03b4 \u00b7 MSE(p, p\u0303\u2033)", fs=14, color=INK, align="left", family=SERIF)
    pill(p, x + 175, Y0 + 456, "weight 0.05 per term \u00b7 both levels", color=CRIM, fill=CRIM_L, fs=12.5, h=28, w=262)
    p.t(x + 20, Y0 + 500, "Important sources must matter; unimportant", fs=12.5, color=MUTE, align="left")
    p.t(x + 20, Y0 + 521, "ones must not.   \u03b7 = \u03b4 = 1 per component.", fs=12.5, color=MUTE, align="left")


def levels(p):
    panel(p, 30, 800, 2340, 300, GREY, "Applied at two levels", tag="same procedure, different graph")
    specs = [
        (885, PURP, "Level 1  \u00b7  AU \u2192 AU", var("V", "a"), "AU\u2013AU graph", "AU j", "AU classifier",
         ["sources i :  every AU, features V<sub>a</sub>", "targets j :  each AU in turn",
          "routing :  \u00c2 from the AU\u2013AU graph (N \u00d7 N)"], 21),
        (1010, CRIM, "Level 2  \u00b7  AU \u2192 Expression", var("U", "a"), "AU\u2192Expr graph", "Expr k",
         "expression classifier",
         ["sources i :  updated AUs, features U<sub>a</sub>", "targets k :  each of the 8 expressions",
          "routing :  \u00c2 expression rows \u00d7 AU columns (8 \u00d7 N)"], 22),
    ]
    for (y, c, title, src, gname, tgt, clf, notes, sd) in specs:
        p.t(64, y - 52, title, fs=15, bold=True, color=c, align="left")
        rows(p, 74, y - 30, 56, 70, 6, BLUE if c == PURP else PURP, seed=sd, pad=4, gap=2)
        p.t(102, y + 54, src, fs=14, color=MUTE)
        p.e([(138, y + 5), (200, y + 5)], color=LINE, w=2.2)
        minigraph(p, 262, y + 5, PURP, seed=sd, n=6, rx=46, ry=26, r=10)
        p.t(262, y + 54, gname, fs=12, color=PURP)
        au = tgt.startswith("AU")
        p.e([(318, y + 5), (378, y + 5)], color=PURP, w=2.2)
        ell(p, 406, y + 5, 44, "#F4D5CB" if au else "#D7E3F7", TERRA if au else "#4F73B5", sw=2.4, value=tgt, fs=11,
            bold=True, color="#7A2F20" if au else "#233E70")
        p.e([(430, y + 5), (480, y + 5)], color=LINE, w=2.2)
        pill(p, 580, y + 5, clf + "  \u2192  " + I("\u0177") + sub("j" if au else "k"), color=INK, fill="#F3F5F8", fs=13,
             h=30, w=190, stroke=GREY_M)
        for i, nline in enumerate(notes):
            p.t(760, y - 20 + i * 26, nline, fs=13, color=INK if i < 2 else MUTE, align="left")
    p.e([(60, 950), (2340, 950)], color=GREY_M, w=1.2, head=None, rounded=False, dashed=True)


def build():
    p = Page("3 \u00b7 Counterfactual", W, H, pid="counterfactual")
    p.t(36, 50, "Counterfactual intervention", fs=34, bold=True, color=INK, align="left")
    p.t(36, 88, "Phases 3\u20134 \u00b7 training-only regularizer: remove the CLIP-defined semantic component from important vs unimportant sources",
        fs=15, color=MUTE, align="left")
    panel(p, 30, 130, 2340, 640, CRIM, "Intervention pipeline", tag="per source AU i \u2192 target j")
    for k in range(6):
        x = X0 + k * (CARD_W + GAP)
        card(p, x, Y0, CARD_W, CARD_H, CRIM, arc=14)
        if k < 5:
            p.e([(x + CARD_W + 3, Y0 + CARD_H / 2), (x + CARD_W + GAP - 3, Y0 + CARD_H / 2)], color=CRIM, w=2.6)
    for f in (s1, s2, s3, s4, s5, s6):
        f(p)
    levels(p)
    return p
